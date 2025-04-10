from graph.graph import Graph
import tqdm

graph = Graph()

import os

file_limit = 5

def extract_txt_files(root_dir):
    txt_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        files_in_dir = 0
        for file in filenames:
            if file.endswith('.txt'):
                full_path = os.path.join(dirpath, file)
                txt_files.append(full_path)
                files_in_dir += 1
                if files_in_dir > file_limit:
                    break

    return txt_files

root_directory = 'process_text/data/arxiv-txt-cs/'  # Replace with your root directory path
txt_file_paths = extract_txt_files(root_directory)

added_edges = set()

import csv

for path in tqdm.tqdm(txt_file_paths):
    with open(path, newline='') as csvfile:
        triplets_reader = csv.reader(csvfile, delimiter=";")
        for triplet in triplets_reader:
            agent_1, action, agent_2 = triplet
            edge = (
                agent_1,
                agent_2,
                action,
            )
            if len(agent_1) == 0 or len(agent_2) == 0 or len(action) == 0:
                continue

            # if edge[0] == "+30" or edge[1] == "+30":
            #     continue

            if "id" in edge[0] or "id" in edge[1] or "id" in edge[2]:
                continue

            if "im" in edge[0] or "im" in edge[1] or "im" in edge[2]:
                continue
            
            if "3mb" in edge[0] or "3mb" in edge[1] or "3mb" in edge[2]:
                continue

            if "10mb" in edge[0] or "10mb" in edge[1] or "10mb" in edge[2]:
                continue

            if "nsga" in edge[0] or "nsga" in edge[1] or "nsga" in edge[2]:
                continue
                
            if "%" in edge[0] or "%" in edge[1] or "%" in edge[2]:
                continue

            if edge[0].count(" ") > 2 or edge[1].count(" ") > 2 or edge[2].count(" ") > 2:
                continue

            if len(edge[0]) < 4 or len(edge[1]) < 4:
                continue

            if edge not in added_edges:
                added_edges.add(edge)
                if not graph.contains_vertex(edge[0]):
                    graph.add_vertex(edge[0])
                if not graph.contains_vertex(edge[1]):
                    graph.add_vertex(edge[1])
                graph.add_edge(*edge)
