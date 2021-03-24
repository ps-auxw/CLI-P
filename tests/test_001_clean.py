import unittest
import sys
from pathlib import Path

class CleanTestEnvironment(unittest.TestCase):

    @classmethod
    def remove_lmdb(cls, path):
        print(f"Removing LMDB database at {path} ...", flush=True)
        for filename in ['data.mdb', 'lock.mdb']:
            filepath = path / filename
            if not filepath.exists():
                continue
            filepath.unlink()
        path.rmdir()

    @classmethod
    def setUpClass(cls):
        try:
            cls.db_paths = []
            for env_name in ['vectors', 'config']:
                path = Path(f"tests/{env_name}.lmdb")
                cls.db_paths.append(path)
                if not path.exists():
                    continue
                cls.remove_lmdb(path)
            for index_name in ['images', 'faces']:
                path = Path(f"tests/{index_name}.index")
                cls.db_paths.append(path)
                if path.exists():
                    path.unlink()
        except BaseException as ex:
            sys.excepthook(type(ex), ex, ex.__traceback__)
            sys.exit(f"{cls.__name__}: Error during pre-tests cleanup, aborting test suite!")

    def run(self, result=None):
        result = super().run(result)
        if len(result.errors) > 0 or len(result.failures) > 0:
            print(f"{self.__class__.__name__}: Pre-tests cleanup post-check failed, stopping test suite!", flush=True)
            result.stop()

    def test_dbs_gone(self):
        for db_path in self.db_paths:
            with self.subTest(db_path=db_path):
                self.assertFalse(db_path.exists(), msg=f"Database file still exists: {db_path}")

if __name__ == '__main__':
    unittest.main()
