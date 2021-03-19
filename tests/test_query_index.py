import unittest
import contextlib
from io import StringIO

class TestQueryIndex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.query_index = __import__("query-index")
        cls.search = cls.query_index.Search()

    @classmethod
    def tearDownClass(cls):
        pass

    @classmethod
    def capture_stdout(cls, code):
        ret = None
        with contextlib.closing(StringIO()) as f:
            with contextlib.redirect_stdout(f):
                ret = code()
            return f.getvalue(), ret

    def verify_line_structure(self, line, msg=None):
        msg_suffix = "no line structure" + ('' if msg is None else ' : ' + msg)
        if line is None:
            self.fail("line is None => " + msg_suffix)
        if not line.endswith("\n"):
            self.fail("line doesn't end with newline => " + msg_suffix)

    def verify_singleline_structure(self, line, msg=None):
        msg_suffix = "no singleline structure" + ('' if msg is None else ' : ' + msg)
        self.verify_line_structure(line, msg_suffix)
        count = line.count("\n")
        if count != 1:
            self.fail(f"{count} newlines instead of 1 => " + msg_suffix)

    def verify_command_behaviour_setfield(self, command_name, field_name,
        value, expect_success, success_msg_prefix=None, fail_msg_prefix=None,
        expect_iterationDone=True, expect_singleline_structure=True,
        interpolate_value=True):
        search = self.search
        setattr(search, field_name, None)
        search.in_text = f"{command_name} {value}" if interpolate_value else command_name
        msg_suffix = f"failed at command {command_name!r} data point {value} expect {'success' if expect_success else 'fail'}"

        output, iterationDone = self.capture_stdout(search.do_command)
        if expect_iterationDone is not None:
            self.assertEqual(expect_iterationDone, iterationDone, msg=f"Command {'did' if iterationDone else 'didnt'} request 'no search'. => {msg_suffix}")

        if expect_singleline_structure is not None:
            if not expect_singleline_structure:
                raise NotImplementedError("expect_singleline_structure=False not implemented => " + msg_suffix)
            self.verify_singleline_structure(output, msg=f"Command didn't return exactly a single line of output: {output!r} => {msg_suffix}")

        if expect_success is None:
            return
        expect_msg_prefix = success_msg_prefix if expect_success else fail_msg_prefix
        if expect_msg_prefix is not None:
            if not output.startswith(expect_msg_prefix):
                self.assertEqual(expect_msg_prefix, output, msg="Output message didn't start with expected message prefix. => " + msg_suffix)

        stored_value = getattr(search, field_name)
        if expect_success:
            self.assertEqual(value, stored_value, msg="Command didn't update Search object to data point value. => " + msg_suffix)
        else:
            self.assertIsNone(stored_value, msg="Command updated Search object although it shouldn't have. => " + msg_suffix)

    def wrap_command_field_reset(self, command_name, field_name, code):
        prev_value = getattr(self.search, field_name)
        try:
            code(command_name, field_name)
        finally:
            setattr(self.search, field_name, prev_value)

    def test_command_q(self):
        self.wrap_command_field_reset("q", "running_cli", lambda command_name, field_name:
            self.verify_command_behaviour_setfield(command_name, field_name, False, True,
                expect_singleline_structure=None, interpolate_value=False)
        )

    def test_command_ft(self):
        values = [
            (1.0, True), (0.5, True), (0.0, True),
            (1.1, False), (-0.1, False),
        ]
        for v in values:
            self.wrap_command_field_reset("ft", "face_threshold", lambda command_name, field_name:
                self.verify_command_behaviour_setfield(command_name, field_name, v[0],  v[1],
                    success_msg_prefix="Set face similarity threshold",
                    fail_msg_prefix="Invalid face threshold")
            )

if __name__ == '__main__':
    unittest.main()
