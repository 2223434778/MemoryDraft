"""Bounded two-GPU extension using the existing immutable-source supervisor."""
from scripts.run_asd_tree_pipeline import main

if __name__=='__main__':
    main(study='asd-paper-extension',worker_module='scripts.run_asd_paper_extension',
        report_module='scripts.write_asd_paper_extension_report',manifest_name='ASD_PAPER_EXTENSION_SOURCE_MANIFEST.json',
        worker_timeout=21600,extra_tests=('tests/test_asd_paper.py','tests/test_asd_paper_extension.py'))
