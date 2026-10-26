"""Bounded two-card DSpark comparison in a new isolated root."""
from scripts.run_asd_tree_pipeline import main

if __name__=='__main__':
    main(study='dspark-tree',worker_module='scripts.run_dspark_tree',
         report_module='scripts.write_dspark_tree_report',manifest_name='DSPARK_TREE_SOURCE_MANIFEST.json',
         extra_tests=('tests/test_dspark.py','tests/test_dspark_report.py'),worker_timeout=10800)
