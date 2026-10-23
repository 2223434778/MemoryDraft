"""Run the frozen joint-allocation study in an isolated source checkout."""
from memorydraft.evaluation.asd_joint_tree import STUDY
from scripts.run_asd_tree_pipeline import main


if __name__ == "__main__":
    main(study=STUDY,worker_module="scripts.run_asd_joint_tree",
         report_module="scripts.write_asd_joint_tree_report",manifest_name="ASD_JOINT_TREE_SOURCE_MANIFEST.json",
         extra_tests=("tests/test_joint_tree.py","tests/test_asd_joint_tree.py"))
