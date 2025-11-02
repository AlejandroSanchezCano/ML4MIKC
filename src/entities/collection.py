
# Built-in modules
from pathlib import Path

# Third-party modules
from tqdm import tqdm

# Custom modules
from src.misc import path
from src.misc import utils
from src.entities.protein import Protein

class Collection:

    def __init__(self, dir: str | Path, class_type: str):
        self.dir = Path(dir)
        self.class_type = class_type
        files = sorted(list(self.dir.glob('*')))
        self._iterator = iter(tqdm(files, desc=f'Loading {class_type} objects'))

    def __iter__(self):
        return self

    def __next__(self):
        file_path = next(self._iterator)
        file_content = utils.unpickle(file_path)
        match self.class_type:
            case 'Protein':
                obj = Protein(**file_content)
            case _:
                raise ValueError(f'Unknown class_type: {self.class_type}')
        return obj


if __name__ == "__main__":
    collection = Collection(dir = path.MIKC_PROTS, class_type = 'Protein')
    for item in collection:
        print(item)
        break