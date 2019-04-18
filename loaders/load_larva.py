""" Janelia Larva TEM data

Some design choices:
1. Neurons default to local neuron. Mostly for those in AL
2. Synapse's neuropil follows presynaptic neuron's neuropil
3. Datasource only owns neuron morphology but not synapse morphology, following current NA schema.
"""
#!/usr/bin/env python

# Copyright (c) 2019, Tingkai Liu
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

import cPickle as pickle
import copy
import logging
import sys

import networkx as nx
from pyorient.ogm import Graph, Config

from neuroarch.models import *
from neuroarch.utils import byteify, chunks, get_cluster_ids

import csv

import pandas as pd
import json
import numpy as np

from ast import literal_eval

from pyorient.serializations import OrientSerialization

def load_swc(file_name):
    """
    Load an SWC file into a DataFrame.
    """

    df = pd.read_csv(file_name, delimiter=' ', header=None, comment='#',
                     names=['sample', 'identifier', 'x', 'y', 'z', 'r', 'parent'],
                     skipinitialspace=True).astype({'sample':int,'identifier':int,'x':float,'y':float,'z':float,'r':float,'parent':int})
    return df

class LarvaLoader(object):
    neuropils = {'AL':('AL',['right antennal lobe','antennal lobe','al_r','al','right al']) ,
           'al':('al',['left antennal lobe','antennal lobe','al_l','al','left al']),
           'MB':('MB',['right mushroom body','mushroom body','right mb','mb_r','mb']),
           'mb':('mb',['left mushroom body','mushroom body','left mb','mb_l','mb']),
           'LON':('LON',['right larva optic neuropil','optic neuropil','right lon','lon_r','lon']),
           'lon':('lon',['left larva optic neuropil','optic neuropil','left lon','lon_f','lon'])
           }

    PN_keys = ['PN','MBON', 'LH-MB','LONI','PVL09','LON-KC','pOLP','LN4','nc2','LN1','LN3','LN2','5thLN','nc1','FLON']
    input_keys = ['KC', 'DAN','OAN','MBIN','ORN']
    LN_keys = ['APL', 'ChalOLP', 'GlulOLP']

    neurotransmitter_map = {
        'KC': 'acetylcholine',
        'ChalOLP': 'acetylcholine',
        'APL': 'GABA',
        'OAN': 'octopamine',
        'DAN': 'dopamine',
        'GlulOLP': 'glutamate'
    }

    # everything else is assumed to be local neuron
    def __init__(self, g_orient):
        self.logger = logging.getLogger('vl')
        self.g_orient = g_orient

        # Make sure OrientDB classes exist:
        #self.g_orient.create_all(Node.registry)
        #self.g_orient.create_all(Relationship.registry)

        # Get cluster IDs:
        self.cluster_ids = get_cluster_ids(self.g_orient.client)

    def load_neurons(self, file_name, conn_file_name, morph_dir):
        ds_fc = self.g_orient.DataSources.query(name='Janelia').first()
        if not ds_fc:
            ds_fc = self.g_orient.DataSources.create(name='Janelia')

        df = pd.read_csv(file_name)
        neuron_nodes_dict = {}
        for i, neuron in df.iterrows():
            print(i,neuron)
            # Process a neuron
            # ['neuron', 'skeleton_id', 'neuropil', 'side', 'name', 'uname']
            neuropil = neuron.neuropil
            side = neuron.side
            if side == 'left':
                neuropil = neuropil.lower()
            elif side== 'right':
                neuropil = neuropil.upper()
            else: # DEBUG: default to right side.
                neuropil = neuropil.upper()

            # Check if neuropil exists
            npl = self.g_orient.Neuropils.query(name=LarvaLoader.neuropils[neuropil]).first()
            if not npl:
                npl = self.g_orient.Neuropils.create( \
                            name=LarvaLoader.neuropils[neuropil],
                            synonyms=LarvaLoader.neuropils[neuropil])
                self.logger.info('created node: {0}({1})'.format(npl.element_type, npl.name))

            if any([k in neuron.name for k in LarvaLoader.LN_keys]):
                locality = True
            elif any([k in neuron.name for k in LarvaLoader.PN_keys + LarvaLoader.input_keys]):
                locality = False
            else:
                locality = True # default to local neuron

            # Create Neuron Node

            n = self.g_orient.Neurons.create(\
                name=neuron.name,
                uname=neuron.uname,
                locality=locality)
            self.logger.info('created node: {0}({1})'.format(n.element_type, n.name))
            neuron_nodes_dict[neuron.uname] = n
            # Create Neurotransmitter Node if required
            nt = None
            neurotransmitter = []
            nt_available_names = LarvaLoader.neurotransmitter_map.keys()
            nt_type = [k in neuron.name for k in nt_available_names]
            if sum(nt_type) > 0:
                neurotransmitter = LarvaLoader.neurotransmitter_map[nt_available_names[nt_type]]
                nt = self.g_orient.NeurotransmitterDatas.create( \
                    name=neuron.uname,
                    Transmitters=neurotransmitter)
                self.logger.info('created node: {0}({1})'.format(nt.element_type, nt.name))
            else:
                neurotransmitter = None

            # Create Morphology Node
            df = load_swc('%s/%s.swc' % (morph_dir, neuron.skeleton_id))
            content = byteify(json.loads(df.to_json()))
            content = {}
            content['x'] = df['x'].tolist()
            content['y'] = df['y'].tolist()
            content['z'] = df['z'].tolist()
            content['r'] = df['r'].tolist()
            content['parent'] = df['parent'].tolist()
            content['identifier'] = df['identifier'].tolist()
            content['sample'] = df['sample'].tolist()

            content.update({'name': neuron.uname })

            nm = self.g_orient.client.record_create(self.cluster_ids['MorphologyData'][0],
                                                    {'@morphologydata': content})
            nm = self.g_orient.get_element(nm._rid)


            # Add content to new node:
            self.g_orient.client.command('update %s content %s' % \
                                        (nm._id, json.dumps(content)))

            self.logger.info('created node: {0}({1})'.format(nm.element_type, nm.name))

            # Connect nodes
            self.g_orient.Owns.create(npl, n)  # neuropil owns neuron
            # self.g_orient.HasData.create(n, arb)  # neuron owns arborization
            self.g_orient.HasData.create(n, nm)  # neuron owns morphological data
            if nt:  # neurotransmitter
                self.g_orient.HasData.create(n, nt)  # neuron owns neurotransmitter data
                self.g_orient.Owns.create(ds_fc, n)  # Datasource owns neuron
            self.g_orient.Owns.create(ds_fc, nm)    # datasource owns the morphologydata
            # self.g_orient.Owns.create(ds_fc, arb)   # datasource owns the arborization

        # TODO:  add synapses
        conn_df = pd.read_csv(conn_file_name)
        for pre_neuron_name in conn_df.presynaptic.unique():
            post_rows = conn_df[conn_df['presynaptic']==pre_neuron_name]
            pre_neuron_node = neuron_nodes_dict[pre_neuron_name]
            for idx, row in post_rows.iterrows():
                syn = self.g_orient.Synapse.create(\
                    name=row.uname,
                    N=row.N,
                    uname=row.uname)
                self.logger.info('created node: {0}({1})'.format(syn.element_type, syn.name))

                content = {}
                content['x'] = literal_eval(','.join(row.x.replace("\n","").replace("  "," ").split(" ")))
                content['y'] = literal_eval(','.join(row.y.replace("\n","").replace("  "," ").split(" ")))
                content['z'] = literal_eval(','.join(row.z.replace("\n","").replace("  "," ").split(" ")))
                content['r'] = literal_eval(','.join(row.r.replace("\n","").replace("  "," ").split(" ")))
                content['parent'] = [-1]*len(content['x'])
                content['identifier'] = [-1]*len(content['x'])
                content['sample'] = list(np.arange(len(content['x'])))
                content.update({'uname': row.uname, 'name':row.uname})
                content['morpho_type'] = 'Synapse SWC'
                syn_nm = self.g_orient.client.record_create(self.cluster_ids['MorphologyData'][0],
                                                            {'@morphologydata': content})
                syn_nm = self.g_orient.get_element(syn_nm._rid)
                self.logger.info('created node: {0}({1})'.format(syn_nm.element_type, syn_nm.name))

                post_neuron_node = neuron_nodes_dict[row.postsynaptic]
                self.g_orient.SendsTo.create(pre_neuron_node, syn)
                self.g_orient.SendsTo.create(syn, post_neuron_node)
                neuropil = df.loc[df['uname']==pre_neuron_name, "neuropil"]
                npl = self.g_orient.Neuropils.query(name=LarvaLoader.neuropils[neuropil]).first()

                self.g_orient.Owns.create(npl, syn)
                self.g_orient.HasData.create(syn, syn_nm)



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    g_orient = Graph(Config.from_url('/na_server_larva','root', 'root', initial_drop=True,
                                     serialization_type=OrientSerialization.Binary))# set to True to erase the database
    g_orient.create_all(Node.registry)
    g_orient.create_all(Relationship.registry)

    vl = LarvaLoader(g_orient)

    vl.load_neurons('all_neurons_reference.csv', 'all_connectors.csv', 'swc')