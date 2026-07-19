import unittest

from src.main import should_train_main


class TrainingScheduleTests(unittest.TestCase):
    def test_reference_schedule_trains_every_ten_rounds_before_1000(self):
        self.assertTrue(should_train_main(10, "reference_training"))
        self.assertFalse(should_train_main(50 + 1, "reference_training"))
        self.assertTrue(should_train_main(1000, "reference_training"))

    def test_default_schedule_is_unchanged(self):
        self.assertTrue(should_train_main(50, "none"))
        self.assertFalse(should_train_main(10, "none"))
        self.assertTrue(should_train_main(2000, "none"))


if __name__ == "__main__":
    unittest.main()
