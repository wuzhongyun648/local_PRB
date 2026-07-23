import unittest

from src.main import (
    ADAPTIVE_DYN_METHOD,
    LOC_METHOD,
    METHOD_ALIASES,
    PRB_METHOD,
    PURE_DYN_METHOD,
    parse_method,
)


class MethodInterfaceTests(unittest.TestCase):
    def test_exactly_four_formal_interfaces(self):
        expected = {
            "PRB",
            "numba-locPRB",
            "numba-adaptive-dyn",
            "numba-pure-dyn",
        }
        self.assertEqual(set(METHOD_ALIASES), expected)
        self.assertEqual(
            {
                PRB_METHOD,
                LOC_METHOD,
                ADAPTIVE_DYN_METHOD,
                PURE_DYN_METHOD,
            },
            expected,
        )

    def test_old_and_python_interfaces_are_rejected(self):
        for method in (
            "LocPRB",
            "dyn_locPRB",
            "dyn-LocPRB",
            "python-locPRB",
            "python-dyn",
        ):
            with self.subTest(method=method):
                with self.assertRaisesRegex(
                    Exception, "unknown method"
                ):
                    parse_method(method)


if __name__ == "__main__":
    unittest.main()
