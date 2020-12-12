from itertools import repeat

from .random import as_random_state
from .random import randint


class Node(object):
    """smaller than namedtuple"""
    __slots__ = 'v', 'left', 'right'

    def __init__(self,
                 v,
                 left=None,
                 right=None):
        """Tree node with either 1 or 2 children"""
        self.v = v
        self.left = left
        self.right = right

    def __str__(self):
        return str(self.v)


class RandExprTree(object):
    __slots__ = 'root'

    def __init__(self,
                 leaves,
                 parents_with_children=None,
                 parents_with_child=None,
                 root_blacklist=None,
                 seed=None):
        """

        :param leaves: values cannot be None or things could break
        :param parents_with_children:
        :param parents_with_child:
        :param root_blacklist:
        :param seed:
        """
        rs = as_random_state(seed)

        if root_blacklist is not None:
            root_blacklist = set(root_blacklist)

        n_leaf = len(leaves)
        if not n_leaf:
            # no nodes
            raise ValueError('Need 1+ leaf nodes')
        if n_leaf == 1:
            # single node
            assert (parents_with_children is None or
                    not len(parents_with_children))
            root_v = leaves[0]
            if root_blacklist and root_v in root_blacklist:
                raise ValueError(
                    f'Cannot have all nodes {root_v} in '
                    f'blacklist_root {root_blacklist}'
                )
            self.root = Node(root_v)
        else:
            # 2+ nodes
            assert (parents_with_children is not None and
                    len(parents_with_children) == (n_leaf - 1))

            # construct tree now
            leaves = [Node(lf) for lf in leaves]
            potential_children = leaves

            # tuples of (nargs, parent)
            parents = [*zip(repeat(2), parents_with_children)]
            if parents_with_child:
                parents.extend(zip(repeat(1), parents_with_child))

            # more randomness
            rs.shuffle(parents)

            # pre-allocate root node value if applicable and possible
            root_v = None
            if root_blacklist:
                choice_idxs = [*range(len(parents))]
                # even more randomness
                rs.shuffle(choice_idxs)

                for idx in choice_idxs:
                    if parents[idx][1] not in root_blacklist:
                        root_v = parents.pop(idx)
                        break
                else:
                    # BIST ;)
                    assert root_v is None
                    raise ValueError(f'Every parent node is in '
                                     f'blacklist_root! {root_blacklist}')

            for n_args, parent in parents:
                # select children
                args = [
                    potential_children.pop(
                        randint(len(potential_children), seed=rs))
                    for _ in range(n_args)
                ]
                potential_children.append(Node(parent, *args))

            if root_v:
                n_args, parent = root_v
                assert len(potential_children) == n_args
                self.root = Node(parent, *potential_children)
            else:
                assert len(potential_children) == 1
                self.root = potential_children[0]

    def __str__(self):
        return tree_to_str(self)

    def to_expression(self):
        pass


def tree_to_str(tree: RandExprTree, padding=2):
    """"""
    lines = _str_tree_helper(((0, tree.root),), padding)
    return '\n'.join(''.join(l) for l in lines[:-1])


def _str_tree_helper(nodes, padding):
    """"""
    children = []
    node_strs = []
    node_ws = []
    all_none = True

    for parent_w, node in nodes:
        if node is None:
            as_str = ''
            node_l_w = node_r_w = padding
            left = right = None
            node_ws.append(node_l_w + node_r_w)
        else:
            as_str = str(node)

            total_width = max(len(as_str) + 2 * padding, parent_w)
            node_l_w = round(total_width / 2)
            node_r_w = total_width - node_l_w

            left = node.left
            right = node.right

            all_none = all_none and left is None and right is None
            node_ws.append(total_width)

        node_strs.append(as_str)
        children.extend(((node_l_w, left), (node_r_w, right)))

    if all_none:
        all_strs = []
        child_strs = repeat(None)
    else:
        all_strs = _str_tree_helper(children, padding=padding)
        child_strs = all_strs[-2]

    out_strs = []
    pretty_strs = []
    out_str = ''
    pretty_str = ''
    is_left = True
    i_node_strs = iter(node_strs)
    i_node_ws = iter(node_ws)
    pretty_flag = True

    for child_str, (node_w, _) in zip(child_strs, children):

        if all_none:
            whitespace = ''
        else:
            whitespace = ' ' * (len(child_str) - node_w)

        if is_left:
            node_str = next(i_node_strs)
            node_w_tot = next(i_node_ws)
            fmt_str = '{:^' + str(node_w_tot) + '}'

            out_str += whitespace + fmt_str.format(node_str)
            pretty_str += ((' ' if pretty_flag else '_') *
                           (len(whitespace) + node_w) + '|')
        else:
            out_str += whitespace
            pretty_str += (('_' if pretty_flag else ' ') *
                           (len(whitespace) + node_w - 1))
            out_strs.append(out_str)
            pretty_strs.append(pretty_str)
            out_str = ''
            pretty_str = ''
            pretty_flag = not pretty_flag

        is_left = not is_left

    all_strs.append(out_strs)
    all_strs.append(pretty_strs)

    return all_strs
