"""Regression checks for the former state entry point's damage-only output."""
import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from scripts.build_sam3_damage_dataset import collect_images, filter_damage, write_sample, yolo_line, plan_append, save_metadata, generate, prompt_thresholds


def candidate(box, score=0.8):
    return SimpleNamespace(box_xyxy=box, score=score, prompt="some SAM3 concept")


class DamageDatasetTests(unittest.TestCase):
    def test_prompt_specific_thresholds_do_not_lower_generic_threshold(self):
        thresholds = prompt_thresholds(None, .45, ['silver ring=0.05', 'silver patch on black pipe=0.10'])
        self.assertEqual(thresholds['damage'], .45)
        self.assertEqual(thresholds['silver ring'], .05)
        self.assertEqual(thresholds['silver patch on black pipe'], .10)
        self.assertEqual(prompt_thresholds(['damage'], .45, ['damage=0.2'])['damage'], .2)
        for invalid in ('cable=0.1', 'silver ring=nan', 'silver ring=0', 'broken'):
            with self.assertRaises(ValueError):
                prompt_thresholds(None, .45, [invalid])

    def test_two_generate_batches_merge_metadata_without_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root/'model.pt'
            checkpoint.touch()
            out = root/'out'
            processor = MagicMock()
            fake_helpers = SimpleNamespace(
                Sam3Processor=MagicMock(return_value=processor),
                build_sam3_image_model=MagicMock(), collect_candidates=lambda *a: [],
                register_dtype_alignment_hooks=lambda model: None,
                resolve_bpe_path=lambda path: root/'vocab.gz')
            for index, confidence in enumerate((0.45, 0.6), start=1):
                source = root/f'batch{index}.jpg'
                cv2.imwrite(str(source), np.zeros((80, 120, 3), dtype=np.uint8))
                args = SimpleNamespace(input=source, out_dir=out, limit=None, append=True,
                                       checkpoint=checkpoint, device='cpu', prompts=None,
                                       bpe_path=None, confidence=confidence, min_area_ratio=.0002,
                                       max_box_area_ratio=.35, max_box_span_ratio=.85,
                                       iou_threshold=.65, save_previews=True)
                with patch.dict('sys.modules', {'torch': SimpleNamespace(),
                                               'scripts.sam3_segment_candidates': fake_helpers}):
                    generate(args)
            metadata = json.loads((out/'annotations.json').read_text())
            self.assertEqual([r['settings']['confidence'] for r in metadata['runs']], [.45, .6])
            self.assertEqual([r['run_id'] for r in metadata['items']], [0, 1])
            self.assertEqual([r['image'] for r in metadata['items']], ['images/0001.jpg', 'images/0002.jpg'])
            self.assertEqual((out/'labels/0002.txt').read_bytes(), b'')

    def test_append_preserves_existing_files_uses_max_and_skips_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root/'out'
            frame = np.zeros((80, 120, 3), dtype=np.uint8)
            old, new = root/'old.jpg', root/'new.jpg'
            for source in (old, new):
                cv2.imwrite(str(source), frame)
            write_sample(old, Path('0113.jpg'), frame, [], out, True)
            metadata = {'nc': 1, 'names': ['damage'], 'items': [
                {'source': str(old), 'image': 'images/0113.jpg', 'label': 'labels/0113.txt'}]}
            save_metadata(out, metadata)
            before = {p: p.read_bytes() for folder in ('images', 'labels', 'previews')
                      for p in (out/folder).iterdir()}
            jobs, merged = plan_append(out, [(old, Path('0001.jpg')), (new, Path('0002.jpg'))], True)
            self.assertEqual(jobs, [(new, Path('0114.jpg'))])
            write_sample(new, jobs[0][1], frame, [candidate([10, 10, 30, 30])], out, True)
            merged['items'].append({'source': str(new), 'image': 'images/0114.jpg', 'label': 'labels/0114.txt'})
            save_metadata(out, merged)
            self.assertEqual(len(json.loads((out/'annotations.json').read_text())['items']), 2)
            self.assertTrue(all(p.read_bytes() == value for p, value in before.items()))
            self.assertEqual(plan_append(out, [(new, Path('0001.jpg'))], True)[0], [])
            manifest_before = (out/'annotations.json').read_bytes()
            # Repeat is a CPU-only no-op, even without checkpoint/device options.
            generate(SimpleNamespace(input=new, out_dir=out, limit=None, append=True))
            self.assertEqual((out/'annotations.json').read_bytes(), manifest_before)
            with self.assertRaises(FileExistsError):
                write_sample(new, Path('0113.jpg'), frame, [], out)
            with self.assertRaisesRegex(ValueError, 'use --append'):
                plan_append(out, [], False)
            (out/'labels/0114.txt').unlink()
            with self.assertRaisesRegex(ValueError, 'disagree'):
                plan_append(out, [], True)

    def test_append_rejects_unrecorded_files_and_wrong_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            metadata = {'nc': 1, 'names': ['damage'], 'items': []}
            save_metadata(out, metadata)
            (out/'images').mkdir()
            (out/'images/0001.jpg').touch()
            with self.assertRaisesRegex(ValueError, 'disagree'):
                plan_append(out, [], True)
            metadata['names'] = ['cable']
            save_metadata(out, metadata)
            with self.assertRaisesRegex(ValueError, 'single-class'):
                plan_append(out, [], True)

    def test_reject_whole_cable_by_box_area_and_span(self):
        local = candidate([40, 10, 80, 30])
        large = candidate([10, 10, 170, 80])
        long_thin = candidate([0, 45, 190, 50])
        faint = candidate([100, 10, 120, 30], 0.1)
        self.assertEqual(filter_damage([large, long_thin, faint, local], 200, 100, .45, .35, .85, .65), [local])

    def test_deduplicate_concepts_and_nested_fragments(self):
        outer = candidate([100, 100, 200, 200])
        inner = candidate([120, 120, 160, 180], .9)
        separate = candidate([300, 100, 350, 150])
        result = filter_damage([inner, outer, outer, separate], 1000, 500, .45, .35, .85, .65)
        self.assertEqual(result, [outer, separate])

    def test_multiple_boxes_only_five_fields_and_original_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.jpg'
            frame = np.zeros((100, 200, 3), dtype=np.uint8)
            cv2.imwrite(str(source), frame)
            boxes = [candidate([20, 10, 60, 30]), candidate([100, 50, 140, 90])]
            out = root / 'out'
            write_sample(source, Path('sample.jpg'), frame, boxes, out, True)
            lines = (out/'labels/sample.txt').read_text().splitlines()
            self.assertEqual(lines, ['0 0.200000 0.200000 0.200000 0.200000', '0 0.600000 0.700000 0.200000 0.400000'])
            self.assertEqual(source.read_bytes(), (out/'images/sample.jpg').read_bytes())
            self.assertEqual(cv2.imread(str(out/'previews/sample.jpg')).shape, frame.shape)
            self.assertFalse((out/'masks').exists())
            self.assertEqual(len(list((out/'images').rglob('*.jpg'))), 1)

    def test_no_damage_writes_empty_label_and_no_preview_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = np.zeros((80, 120, 3), dtype=np.uint8)
            source = root/'input.png'
            cv2.imwrite(str(source), frame)
            write_sample(source, Path('group/negative.jpg'), frame, [], root/'out')
            self.assertEqual((root/'out/labels/group/negative.txt').read_bytes(), b'')
            self.assertEqual(cv2.imread(str(root/'out/images/group/negative.jpg')).shape, frame.shape)
            self.assertFalse((root/'out/previews').exists())

    def test_recursive_images_have_stable_unique_sequential_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for group in ('a', 'b'):
                (root/'input'/group).mkdir(parents=True)
                (root/'input'/group/'same.jpg').touch()
            jobs = collect_images(root/'input', root/'out')
            self.assertEqual([p.as_posix() for _, p in jobs], ['0001.jpg', '0002.jpg'])
            (root/'input/a/same.png').touch()
            jobs = collect_images(root/'input', root/'out')
            self.assertEqual([p.name for _, p in jobs], ['0001.jpg', '0002.jpg', '0003.jpg'])
            self.assertEqual([p.parent.name for p, _ in jobs], ['a', 'a', 'b'])
            self.assertEqual(collect_images(root/'input', root/'out', limit=2), jobs[:2])
            self.assertEqual(collect_images(root/'input/b/same.jpg', root/'out')[0][1], Path('0001.jpg'))
            with self.assertRaisesRegex(ValueError, 'outside'):
                collect_images(root/'input', root/'input/out')

    def test_invalid_box_does_not_become_invalid_yolo_label(self):
        with self.assertRaises(ValueError):
            yolo_line([0, 0, 201, 100], 200, 100)


if __name__ == '__main__':
    unittest.main()
