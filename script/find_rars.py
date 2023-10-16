from pathlib import Path
from mirok.absrar_base import AbsRARBase

from mirok.logging_util import init_log
from mirok.rar_finder import RARFinder

if __name__ == '__main__':
    init_log("log/find_rars.log")

    lib_list = [("guava", "libs/guava-32.1.3-jre.sources")] # [(lib_name1, lib_path1), (lib_name1, lib_path1), ...]

    abs_rar_base = AbsRARBase.from_csv("resources/valid_abs_rars.csv")
    finder = RARFinder(abs_rar_base)
    rars = finder.find(lib_list)

    with Path("resources/rars.txt").open("w") as f:
        f.write("\n".join(f"{api1}, {api2}" for api1, api2 in rars))