import os
from open_relation1.vrd_data_config import vrd_predicate_config

# ATTENTION: THIS IS DEPRECATED
# USE: open_relation1.dataset.vrd.label_hier.pre_hier.prenet

class PreNode:
    def __init__(self, name):
        self._name = name
        self._hypers = []

    def __str__(self):
        return self._name

    def name(self):
        return self._name

    def hypers(self):
        return self._hypers

    def append_hyper(self, hyper):
        self._hypers.append(hyper)

    def hyper_paths(self):
        if len(self._hypers) == 0:
            # root
            return [[self]]
        else:
            paths = []
            for hyper in self._hypers:
                sub_paths = hyper.hyper_paths()
                for sub_path in sub_paths:
                    sub_path.append(self)
                    paths.append(sub_path)
            return paths

    def show_hyper_paths(self):
        paths = self.hyper_paths()
        for p in paths:
            str = []
            for n in p:
                str.append(n._name)
                str.append('->')
            str = str[:-1]
            print(' '.join(str))


class PreNet:

    def get_all_labels(self):
        return sorted(self._nodes.keys())

    def get_raw_labels(self):
        return self._raw_labels

    def get_node_by_name(self, name):
        if name in self._nodes:
            return self._nodes[name]
        else:
            return None

    def _load_raw_label(self, raw_label_path):
        labels = []
        if os.path.exists(raw_label_path):
            with open(raw_label_path, 'r') as f:
                raw_lines = f.readlines()
                for line in raw_lines:
                    labels.append(line.strip('\n'))
        else:
            print('raw label file not exists !')
        return labels

    def _construct_hier(self):
        # root node
        root = PreNode('predicate')
        self._nodes['predicate'] = root

        # abstract level
        abs_labels = ['act', 'spa', 'ass', 'cmp']
        for abs_label in abs_labels:
            node = PreNode(abs_label)
            node.append_hyper(root)
            self._nodes[abs_label] = node

        # basic level
        act_labels = [  'wear',    'sleep',    'sit',      'stand',
                        'park',    'walk',     'hold',     'ride',
                        'carry',   'look',     'use',      'cover',
                        'touch',   'watch',    'drive',    'eat',
                        'lying',   'pull',     'talk',     'lean',
                        'fly',     'face',     'rest',     'skate',
                        'follow',  'hit',      'feed',     'kick',
                        'play with']

        spa_labels = [  'on',      'next to',  'above',    'behind',
                        'under',   'near',     'in',       'below',
                        'beside',  'over',     'by',       'beneath',
                        'on the top of',        'in the front of',
                        'on the left of',       'on the right of',
                        'at',      'against',  'inside',   'adjacent to',
                        'across',  'outside of' ]

        ass_labels = [  'has',     'with',     'attach to',    'contain']

        cmp_labels = [  'than']

        basic_label_lists = [act_labels, spa_labels, ass_labels, cmp_labels]
        for i in range(len(abs_labels)):
            abs_label = abs_labels[i]
            abs_pre = self._nodes[abs_label]
            basic_labels = basic_label_lists[i]
            # link basic level to abstract level
            for basic_label in basic_labels:
                pre = PreNode(basic_label)
                pre.append_hyper(abs_pre)
                self._nodes[basic_label] = pre

        # concrete level
        for raw_label in self._raw_labels:
            if raw_label not in self._nodes:
                # predicate phrase
                node = PreNode(raw_label)
                first_space_pos = raw_label.find(' ')
                if first_space_pos == -1:
                    # print('<%s> Not a phrase !!!' % raw_pre)
                    exit(-1)
                phrase = [raw_label[:first_space_pos], raw_label[first_space_pos+1:]]
                for part in phrase:
                    if part in self._nodes:
                        hyper = self._nodes[part]
                        node.append_hyper(hyper)
                    else:
                        # print(' <%s> -> <%s> miss' % (raw_pre, part))
                        pass
                self._nodes[raw_label] = node

    def __init__(self):
        raw_label_path = vrd_predicate_config['raw_label_list']
        self._raw_labels = self._load_raw_label(raw_label_path)
        # action, spatial, association, comparison
        self._nodes = dict()
        self._construct_hier()



#
# if __name__ == '__main__':
#     a = PreNet()
#     n = a.get_node_by_name('stand next to')
#     n.show_hyper_paths()