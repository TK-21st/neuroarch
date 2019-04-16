#!/usr/bin/env python

"""
Detect differences between an original and modified graph represented as Pandas
DataFrames.
"""

# Copyright (c) 2015, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

import re

import daff
import pandas as pd


def diff_edges(old, new, full_replace=True):
    return take_diff('edge', old, new, full_replace)


def diff_nodes(old, new, full_replace=True):
    return take_diff('node', old, new, full_replace)


def take_diff(diff_type, old, new, full_replace=True):
    """
    Find differences between original and modified DataFrame instances.

    Parameters
    ----------
    old, new : pandas.DataFrame
         DataFrame instances to compare.
    full_replace: bool, optional
         set to `False` to update only changed fields in row

    Results
    -------
    result : dict
         Dict containing keys 'add', 'mod', and 'del';
         `result['add']` contains IDs (key) and records (value) to add to `old`;
         `result['mod']` contains IDs (key) in `old` to update with
         corresponding records (value); `result['del']` contains IDs to delete
         from `old`.
    """

    assert isinstance(old, pd.DataFrame) and \
        isinstance(new, pd.DataFrame)

    # Move indices into first column and diff:
    if diff_type == 'edge':
        old.index = old['out'].rename('id').str.cat(
            old['class'], sep=' ').str.cat(old['in'], sep=' ')
        new.index = new['out'].rename('id').str.cat(
            new['class'], sep=' ').str.cat(new['in'], sep=' ')
        new = new.drop_duplicates()

    # Ensure that ids in each DataFrame index are unique:
    assert len(old.index) == len(set(old.index))
    assert len(new.index) == len(set(new.index))

    # Move column names into first row:
    old = pd.concat((pd.DataFrame([old.columns], columns=old.columns),
                        old.copy()))
    new = pd.concat((pd.DataFrame([new.columns], columns=new.columns),
                        new.copy()))

    data = daff.Coopy.diff(old.reset_index().values,
                            new.reset_index().values).data
    df = pd.DataFrame(data)
    # General algorithm as to how to process output of diff (`df`):
    # '->' in first col but not in second col = modify row
    # with second col as ID
    # '->' in first col and in second col = remove row
    # with ID before '->' in second col and add row with
    # ID after '->' in second col
    # '+' in first col = modify row with second col as ID
    # '+++' in first col = add row with second col as ID
    # '---' in first col = remove row with second col as ID
    # '---' or '(.*)' in the first row = modify all entries not already
    # processed

    # Entries to modify (key = ID, value = new attributes):
    mod_dict = dict()

    # Entries to delete:
    del_dict = dict()

    # If there is a column-wide diff operation, the @@ will appear in the second
    # row of the diff output:
    add_dict = dict()

    if df.iloc[0, 0] == '@@':
        colname_row = 0
        start_row = 1
        if len(df) == 1:  # no change
            return {'mod': mod_dict, 'del': del_dict, 'add': add_dict}
    else:
        edit_row = 0
        colname_row = 1
        start_row = 2
    
    # case for only column rename and delete
    if df.iloc[0,0] =='!' and len(df)==2:
        if full_replace:
            mod_dict = new.iloc[1:].to_dict(orient='index')
        else:
            # cols with no-ops in new df
            noop_cols = [df.iloc[1,i+2] for i,k in enumerate(df.iloc[0][2:].values) if len(k)==0]
            op_cols = list(set(new.iloc[0,:].values) - set(noop_cols))
            mod_dict = new.iloc[1:,:]
            mod_dict = mod_dict[op_col].to_dict(orient='index')
        return {'mod': mod_dict, 'del': del_dict, 'add': add_dict}


    print(df)
    print(new)
    print(old)
    # Entries to add:
    for i in xrange(start_row, len(df)):
        op = df.iloc[i][0]
        id = df.iloc[i][1]
        print(op,id, {'add':add_dict,'mod': mod_dict,'del': del_dict})
        print('-----------------')
        if op == '...' or op == '':
            continue
        elif op == '+++':
            add_dict[id] = new.loc[id].to_dict()
        elif op == '---':
            del_dict[id] = None
            continue
        elif op == '+':
            if full_replace:
                mod_dict[id] = new.loc[id].to_dict()
        elif '->' in op:
            # Renaming an ID requires deletion of the original row and creation
            # of a new row:
            if isinstance(id, str) and op in id:
                # Extract ID from 'old->new':
                old_id, new_id = re.search('(.+)%s(.+)$' % op, id).groups()

                # Preserve type of new row ID:
                old_id = old.index.dtype.type(old_id)
                new_id = new.index.dtype.type(new_id)

                del_dict[old_id] = None
                add_dict[new_id] = new.loc[new_id].to_dict()
            else:
                # If full_replace then update all fields for rows that have any change,
                # else only update the fields with changes
                if full_replace:
                    mod_dict[id] = new.loc[id].to_dict()
                else:
                    if not id in mod_dict:
                        mod_dict[id] = dict()
                    for row_val, col in zip(df.iloc[i][1:], df.iloc[colname_row][1:]):
                        if '->' in str(row_val):
                            mod_dict[id][col] = new.loc[id][col]
        # cols that have --- or +++ need to have all fields removed/added
        if df.iloc[0, 0] == '!':
            for col_val, col in zip(df.iloc[edit_row][1:], df.iloc[colname_row][1:]):
                print('haha:::',id,col_val,col)
                if id not in mod_dict and id not in del_dict:
                    mod_dict[id] = dict()
                if '+++' in str(col_val):
                    if op != '---':
                        mod_dict[id][col] = new.loc[id][col]
                elif '---' in str(col_val):
                    mod_dict[id][col] = None

    return {'mod': mod_dict, 'del': del_dict, 'add': add_dict}
