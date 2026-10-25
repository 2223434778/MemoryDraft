"""Isolated continuation; original failed root and archive remain immutable."""
from scripts.run_asd_tree_pipeline import main

if __name__=='__main__':
    main(study='asd-paper-extension',worker_module='scripts.run_asd_paper_extension_continuation',
        report_module='scripts.write_asd_paper_extension_continuation_report',
        manifest_name='ASD_PAPER_EXTENSION_CONTINUATION_SOURCE_MANIFEST.json',worker_timeout=21600,
        extra_tests=('tests/test_asd_paper.py','tests/test_asd_paper_extension.py','tests/test_asd_extension_resume.py'))
