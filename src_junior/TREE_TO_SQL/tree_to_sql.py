"""Convert Decision tree to JSON format"""
from typing import List
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

def is_node_leaf(node_index: int, tree: DecisionTreeClassifier) -> bool:
    """Checking whether a tree node is a leaf or not"""
    is_leaf = tree.tree_.children_left[node_index] == -1 and\
              tree.tree_.children_right[node_index] == -1
    return is_leaf

def convert_tree_to_json(tree: DecisionTreeClassifier,
                         node_index=0,
                         json_dict=None) -> str:
    """Converting a Decision Classifier tree into JSON"""
    json_dict = {}

    # checking if the current node is a leaf
    if is_node_leaf(node_index, tree):
        # class label for the leaf
        class_label = int(tree.tree_.value[node_index].argmax())
        json_dict['class'] = class_label
    else:
        # division feature index
        feature_index = int(tree.tree_.feature[node_index])
        json_dict['feature_index'] = feature_index

        # corresp. threshold
        threshold = round(float(tree.tree_.threshold[node_index]), 4)
        json_dict['threshold'] = threshold

        # recursion step for the left and right children nodes
        left_child_index = tree.tree_.children_left[node_index]
        json_dict['left'] = convert_tree_to_json(tree, left_child_index)

        right_child_index = tree.tree_.children_right[node_index]
        json_dict['right'] = convert_tree_to_json(tree, right_child_index)

    # when the recursion is finished, convert dict into str
    if node_index == 0:
        json_str = json.dumps(json_dict)
        return json_str

    return json_dict


def generate_sql_query(tree_as_json: str, features: list, sub=False) -> str or List:
    """Genetating JSON tree into SQL query"""
    # initialize the var if it's the initial recursion step
    # otherwise, leave it as input
    query = []

    # tree: json into dict
    tree_as_dict = json.loads(tree_as_json)
    # end value as a classifier prediction
    if 'class' in tree_as_dict:
        # class label for the leaf
        class_label = tree_as_dict['class']
        # supplement the query
        query.append(str(class_label))
    else:
        # division feature index
        feature_index = tree_as_dict['feature_index']
        # corresp. threshold
        threshold = tree_as_dict['threshold']
        # supplement the query
        query.append(f'CASE WHEN {features[feature_index]} > {threshold} THEN')

        # recursion step for the right and left children nodes
        right_child_dict = tree_as_dict['right']
        right_sub_query = generate_sql_query(json.dumps(right_child_dict), features, sub=True)
        query.extend(right_sub_query)
        query.append('ELSE')

        left_child_dict = tree_as_dict['left']
        left_sub_query = generate_sql_query(json.dumps(left_child_dict), features, sub=True)
        query.extend(left_sub_query)
        query.append('END')

    # when the recursion is finished, convert the list of query parts into one str
    if not sub:
        query.append('AS class_label')
        query_str = 'SELECT ' + ' '.join(query)
        return query_str

    return query
