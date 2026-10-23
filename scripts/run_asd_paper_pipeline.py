"""Isolated paired study; the post-worker audit uses a local model only."""
from scripts.run_asd_tree_pipeline import main

if __name__=='__main__':
    main(study='asd-paper',worker_module='scripts.run_asd_paper',report_module='scripts.write_asd_paper_report',
        manifest_name='ASD_PAPER_SOURCE_MANIFEST.json',worker_timeout=14400,
        post_worker_module='scripts.judge_asd_paper',extra_tests=('tests/test_asd_paper.py','tests/test_joint_tree.py'))
