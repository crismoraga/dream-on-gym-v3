# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:10:30 2022

@author: redno
"""
import os
import json
import jsonschema
from jsonschema import validate
from ..node import Node
from ..link import Link

class Reader:
    def load_schema(self):
        """Load the JSON schema at the given path as a Python object.
    
        Args:
            schema_path: A filename for a JSON schema.
    
        Returns:
            A Python object representation of the schema.
    
        """
        schema_path = os.path.join(os.path.dirname(__file__), 'bitRates.schema.json')
        try:
            with open(schema_path) as schema_file:
                schema = json.load(schema_file)
        except ValueError as e:
            raise 'Invalid JSON in schema or included schema: %s\n%s' % (schema_file.name, str(e))

        return schema 
    
    def validateJson(self,jsonData):
        """Validate the JSON with a JSON Schema

        Args:
            jsonData: JSON Content how a String.
        """
        try:
            localSchema = self.load_schema()
            validate(instance=jsonData, schema=localSchema)
        except jsonschema.exceptions.ValidationError as err:
            return False
        return True
    
    def readNetwork(self, file : str, nodes, links):
        """Read the file of network and build nodes and links.
        Args:
            file : Filename and path that have the information of Network (Nodes and Links)
            nodes: Free List to add the nodes.
            links: Free list to add the links.
        """
        with open(file) as json_file:
            info = json.load(json_file)
            if (self.validateJson(info)):
                #Carga archivo con slots por bandas
                for readNode in info['nodes']:
                    node = Node(readNode['id'])
                    nodes.append(node)
                for readLink in info['links']:
                    link = Link(
                        readLink['id'], readLink['length'], bands=readLink['slots'])
                    link.src = readLink['src']
                    link.dst = readLink['dst']
                    links.append(link)
            else:
                for readNode in info['nodes']:
                    node = Node(readNode['id'])
                    nodes.append(node)
                for readLink in info['links']:
                    link = Link(
                        readLink['id'], readLink['length'], slots=readLink['slots'])
                    link.src = readLink['src']
                    link.dst = readLink['dst']
                    links.append(link)


