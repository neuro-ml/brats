import torch.nn as nn

from brats.layers import compose_blocks, SplitCat


def check_structure(structure):
    assert all([len(level) == 3 for level in structure[:-1]])
    assert len(structure[-1]) == 2

    for i, line in enumerate(structure):
        assert len(line[0]) > 0
        line[1] = [line[0][-1], *line[1]]
        if i != len(structure) - 1:
            assert line[1][-1] + structure[i + 1][-1][-1] == line[2][0]


def _build_tnet(structure, cb, up, down):
    line, *down_structure = structure
    if len(down_structure) == 0:
        return nn.Sequential(down(), compose_blocks(sum(line, []), cb), up())
    else:
        down_path = build_tnet(down_structure, cb, up, down)
        inner_path = compose_blocks([line[0][-1], *line[1]], cb)
        return nn.Sequential(compose_blocks(line[0], cb),
                             SplitCat(down_path, inner_path),
                             compose_blocks(line[2], cb))


def build_tnet(structure, cb, up, down):
    check_structure(structure)
    return _build_tnet(structure, cb, up, down)
