import base64
import io
import re
from urllib import response
from flask import Flask, request, jsonify,render_template,session,send_from_directory,url_for
import cv2
import numpy as np
import pandas as pd
import os
import ast
from flask_cors import CORS  
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

app = Flask(__name__,static_url_path='/static',static_folder='static')
#app = Flask(__name__)
app.secret_key = 'abcd'

#ground floor graph
edges = [(3, 300, 1), (300, 4, 2), (4, 400, 1), (400, 8, 2), (8, 9, 2), (8, 10, 2), (8, 7, 2),
         (9, 10, 2), (9, 7, 2), (10, 7, 2), (7, 6, 2), (6, 5, 2), (5, 50, 1),(9,900,1),(900,24,2),(24,25,1),(25,26,1),(26,27,1),(27,28,1),(28,29,1),(29,30,1),(30,31,1),(31,32,1),(32,33,1),(31,33,1),(27,29,1),(28,30,1),(31,310,1),(33,310,1),(32,310,1),(1,28,1),(2,29,1),(27,30,1),(310,10,1)]
graph = nx.Graph()
graph.add_weighted_edges_from(edges)
edge_index = torch.tensor(list(graph.edges)).t().contiguous()
edge_attr = torch.tensor([graph[u][v]['weight'] for u, v in graph.edges], dtype=torch.float32).view(-1, 1)
x = torch.randn(1000, 1)  
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

#13th floor graph 
graph_2 = nx.Graph()
edges_2 = [(16,17,1),(17,170,1),(190,19,1),(190,170,1),(170,18,1),(191,19,1),(191,20,1),(20,15,1),(20,22,1),(22,23,1),(22,110,1),(180,18,1),(180,120,1),(130,13,1),(131,13,1),(190,19,1),(14,18,1),(18,21,1),(11,110,1),(12,120,1),(120,130,1),(131,110,1)]
graph_2.add_weighted_edges_from(edges_2)
edge_index_2 = torch.tensor(list(graph_2.edges)).t().contiguous()
edge_attr_2 = torch.tensor([graph_2[u][v]['weight'] for u, v in graph_2.edges], dtype=torch.float32).view(-1, 1)
x = torch.randn(1000, 1)  
data = Data(x=x, edge_index=edge_index_2, edge_attr=edge_attr_2)
# Define a simple GNN model
class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x
model = GNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loader = DataLoader([data], batch_size=1, shuffle=True)
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.x)  
        loss.backward()
        optimizer.step()
        
        
tag_number = None
selected_tag = None
CORS(app) 
csv_file_path = '../capstone_flaskapp/static/newnewORB1.csv'  
df = pd.read_csv(csv_file_path)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


@app.route('/')
def index():
    return render_template('index.html')
'''
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')
'''
@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})
    uploaded_image = request.files['image']
    if not uploaded_image.filename.endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'error': 'Invalid image format'})
    uploaded_image_data = uploaded_image.read()
    uploaded_image_cv2 = cv2.imdecode(np.frombuffer(uploaded_image_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(uploaded_image_cv2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(50)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    descriptors_umat = cv2.UMat(np.array(descriptors, dtype=np.uint8))
    best_match_index = None
    best_match_distance = float('inf')
    for index, row in df.iterrows():
        csv_descriptors = row['descriptors']
        csv_descriptors_list = ast.literal_eval(csv_descriptors)
        csv_descriptors_umat = cv2.UMat(np.array(csv_descriptors_list, dtype=np.uint8))
        matches = bf.match(csv_descriptors_umat, descriptors_umat)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > 0 and matches[0].distance < best_match_distance:
            best_match_index = index
            best_match_distance = matches[0].distance
    if best_match_index is not None:
        best_match_image_path = df.loc[best_match_index, 'Image_Path']
        pattern = r'Tag_(\d+)'
        file_path =best_match_image_path
        match = re.search(pattern, file_path)
        tag_number = None
        tag_name = ""
        if match:
            tag_number = match.group(1)
        if(tag_number == '4'):
            tag_name = "Staff Lift"
        elif(tag_number == '300'):
            tag_name = "G04 Hallway"
        
        elif(tag_number == '40'):
            tag_name = "BG03 Entry"
        elif(tag_number == '3'):
            tag_name = "G04 Lecture Hall"
        elif(tag_number == '5'):
            tag_name = "Ground Floor Staff Room"
        elif(tag_number == '6'):
            tag_name = "G03 Hallway"
        elif(tag_number == '7'):
            tag_name = "G02 Lecture Hall"
        elif(tag_number == '8'):
            tag_name = "Staff Room Office & Bulletin Board"
        elif(tag_number == '9'):
            tag_name = "G05 Lecture Hall"
        elif(tag_number == '10'):
            tag_name = "G01 Research Laboratory"
        elif(tag_number == '50'):
            tag_name = "Chairperson Office"
        elif(tag_number == '1'):
            tag_name = "G08 Lecture Hall"

        elif(tag_number == '2'):
            tag_name = "G10 Laboratory"   

        elif(tag_number == '24'):
            tag_name = "Electrical Room & GU2 Utility"

        elif(tag_number == '25'):
            tag_name = "Girls Toilet"

        elif(tag_number == '26'):
            tag_name = "BGA1 Auditorium"

        elif(tag_number == '27'):
            tag_name = "G06 Lecture Hall"

        elif(tag_number == '28'):
            tag_name = "G07 Lecture Hall"
        
        elif(tag_number == '29'):
            tag_name = "G11 Lecture Hall"

        elif(tag_number == '30'):
            tag_name = "G12 Lecture Hall"

        elif(tag_number == '31'):
            tag_name = "Lobby"

        elif(tag_number == '310'):
            tag_name = "Lobby"

        elif(tag_number == '32'):
            tag_name = "Lift 1"

        elif(tag_number == '33'):
            tag_name = "Lift 2"
            
        elif(tag_number == '11'):
            tag_name = "Backdoor Seating Area 1"
        elif(tag_number == '12'):
            tag_name = "Backdoor Seating Area 2"
        elif(tag_number == '13'):
            tag_name = "Outdoor Exit"
        elif(tag_number == '14'):
            tag_name = "Silent Room"
        elif(tag_number == '15'):
            tag_name = "Outdoor Seating Area"
        elif(tag_number == '16'):
            tag_name = "Canteen"

        elif(tag_number == '17'):
            tag_name = "Canteen Entrance"   

        elif(tag_number == '18'):
            tag_name = "Silent Room Hallway"

        elif(tag_number == '19'):
            tag_name = "13th Floor Entrance"

        elif(tag_number == '20'):
            tag_name = "Outdoor Seating Hallway"

        elif(tag_number == '21'):
            tag_name = "Silent Room Outdoor Area 1"

        elif(tag_number == '22'):
            tag_name = "Silent Room Outdoor Area 2"
        
        elif(tag_number == '23'):
            tag_name = "Silent Room Outdoor Area 3"

        else:
            tag_name = 'No matching training image found'
        result = f'You are currently in : {tag_name}'
        session['tag_number'] = tag_number
    
    return jsonify({'result': result , 'tag_number': tag_number})

@app.route('/set_selected_tag', methods=['POST'])
def set_selected_tag():
    tag_number = session.get('tag_number')
    print("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",tag_number)    #tagnumber is the source locations
    selected_tag = int(request.form.get('selected_tag'))
    img_path = request.form.get('image_path')
    match = re.search(r'\d+', img_path)
    print("se_tag",selected_tag)
    print("img_path",tag_number)
    n = int(tag_number)
    if (n>=11 and n<=23) and (selected_tag<11 or selected_tag>23):
        shortest_path = nx.shortest_path(graph_2, source=n, target=19, weight='weight')
        i_path = '../capstone_flaskapp/static/blueprint_13.jpg' 
        image = plt.imread(i_path)
        shortest_path_graph = graph_2.subgraph(shortest_path)
        pos = {
            11:[5.5,3],
            110:[4.9,3],
            12:[3.2,3],
            120:[3.9,3],
            13:[4.4,3.5],
            130:[3.9,3.5],
            131:[4.9,3.5],
            14: [3.18,2.00],
            15: [5.4,1.00],
            16: [2.5,1.00],
            17: [3.35,1],
            170:[3.9,1],
            18:[3.9,2],
            180:[3.9,2.5],
            19:[4.5,0.3],
            190:[3.9,0.3],
            191:[4.9,0.3],
            20:[4.9,1],
            21:[4.4,2],
            22:[4.9,2],
            23:[5.45,2]
    
        # Define node position
        }

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(image, extent=[0, 8, 0, 4])
        # nx.draw(graph, pos, with_labels=True, node_color='lightgray', node_size=150, ax=ax)
        
        nx.draw(graph_2, pos, with_labels=True, edge_color='black', node_color='lightgray',width = 0.01, node_size=250, ax=ax)

        #icon for source node
        source_node = n
        source_image = plt.imread('../capstone_flaskapp/static/source_node_icon.png')
        imagebox = OffsetImage(source_image, zoom=0.13)
        ab = AnnotationBbox(imagebox,(pos[source_node][0], pos[source_node][1]), frameon=False, boxcoords="data", pad=0)
        ax.add_artist(ab)
        #icon for destination node
        destination_node = 19  # Assuming 'm' is the destination node index
        destination_image = plt.imread('../capstone_flaskapp/static/destination_node_icon.png')
        imagebox_dest = OffsetImage(destination_image, zoom=0.03)
        ab_dest = AnnotationBbox(imagebox_dest, (pos[destination_node][0], pos[destination_node][1]+0.22), frameon=False, boxcoords="data", pad=0)
        ax.add_artist(ab_dest)

        # Draw only the edges along the shortest path with red color
        edge_colors = ['#5bff03' if edge in shortest_path_graph.edges else 'black' for edge in graph_2.edges]
        nx.draw(shortest_path_graph, pos, with_labels=False, node_color='red', node_size=250, ax=ax)
        nx.draw_networkx_edges(graph_2, pos, edgelist=graph_2.edges, edge_color=edge_colors,width =0.01, ax=ax)
        nx.draw_networkx_edges(shortest_path_graph, pos, edgelist=shortest_path_graph.edges,edge_color='#5bff03', width=6.0, ax=ax)
        ax.set_xlim(0, 8)  
        ax.set_ylim(0, 4)  
        ax.axis('on')  
        # plt.grid(True) 
        # for x in range(9):
        #     for y in range(5):
        #         plt.text(x, y, f'({x:.1f}, {y:.1f})', ha='center', va='center', color='black', fontsize=8)
       
        img_path = '../capstone_flaskapp/static/blueprint_13.jpg' 
        plt.savefig(img_path,bbox_inches='tight', pad_inches=1)
        plt.close()
        
        
        #ground floor
        shortest_path = nx.shortest_path(graph, source=32, target=selected_tag, weight='weight')
        i_path = '../capstone_flaskapp/static/shortest_path_image_1.png' 
        image = plt.imread(i_path)
        shortest_path_graph = graph.subgraph(shortest_path)
        pos = {
            1: [7.5,1.95],
            2: [7.5,1],
            3:[0.83,3.5],
    300: [0.83, 2.6],
   4: [1.2, 2.6],
    400: [1.2, 1.95],
    8: [1.79, 1.95],
    9: [2.5, 1.95],
    7: [1.79, 1],
    10: [2.5, 1],
    6: [1.2, 1],
    5: [0.8, 1],
    50: [0.8, 0.25],
    900: [3.2,1.95],
    24: [3.2,3.04],
    25: [4.75,3.04],
    26: [4.75,1.95],
    27: [5.22,1.95],
    28: [6.1,1.95],
    29: [6.1,1],
    30: [5.22,1],
    31: [4.32,1],
    32: [4.32,0.6],
    33: [3.5,0.6],
    310: [3.5,1]
        # Define node positions
        }

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(image, extent=[0, 8, 0, 4])
       
        
        nx.draw(graph, pos, with_labels=True, edge_color='black', node_color='lightgray',width = 0.01, node_size=250, ax=ax)

# Draw only the edges along the shortest path with red color
        edge_colors = ['#5bff03' if edge in shortest_path_graph.edges else 'black' for edge in graph.edges]
        
        nx.draw(shortest_path_graph, pos, with_labels=False, node_color='red', node_size=250, ax=ax)
        
        
        #icon for source node
        source_node = 32
        source_image = plt.imread('../capstone_flaskapp/static/source_node_icon.png')
        imagebox = OffsetImage(source_image, zoom=0.13)
        ab = AnnotationBbox(imagebox,(pos[source_node][0], pos[source_node][1]), frameon=False, boxcoords="data", pad=0)
        ax.add_artist(ab)
        #icon for destination node
        destination_node = selected_tag  # Assuming 'm' is the destination node index
        destination_image = plt.imread('../capstone_flaskapp/static/destination_node_icon.png')
        imagebox_dest = OffsetImage(destination_image, zoom=0.03)
        ab_dest = AnnotationBbox(imagebox_dest, (pos[destination_node][0], pos[destination_node][1]+0.22), frameon=False, boxcoords="data", pad=0)
        ax.add_artist(ab_dest)

        # nx.draw_networkx_nodes(graph, pos, nodelist=[source_node], node_color='#f3f315', node_size=650, ax=ax)
        nx.draw_networkx_edges(graph, pos, edgelist=graph.edges, edge_color=edge_colors,width =0.01, ax=ax)
        nx.draw_networkx_edges(shortest_path_graph, pos, edgelist=shortest_path_graph.edges,edge_color='#5bff03', width=6.0, ax=ax)
        ax.set_xlim(0, 8)  
        ax.set_ylim(0, 4)  
        ax.axis('on')  
        # plt.grid(True) 
        # for x in range(9):
        #     for y in range(5):
        #         plt.text(x, y, f'({x:.1f}, {y:.1f})', ha='center', va='center', color='black', fontsize=8)
        img_path = '../capstone_flaskapp/static/uploads/shortest_path_image.png' 
        plt.savefig(img_path,bbox_inches='tight', pad_inches=1)
        plt.close()
        return jsonify({'image_url_13': '/static/uploads/shortest_path_image.png','image_url': '/static/uploads/blueprint_13.jpg','p1':"Proceed from your current position to Tag 19, and once there, take the elevator to the ground floor."})
        
    elif (n<11 or n>23) and (selected_tag>=11 and selected_tag<=23):
        shortest_path = nx.shortest_path(graph, source=n, target=32, weight='weight')
        i_path = '../capstone_flaskapp/static/shortest_path_image_1.png' 
        image = plt.imread(i_path)
        shortest_path_graph = graph.subgraph(shortest_path)
        pos = {
            1: [7.5,1.95],
            2: [7.5,1],
            3:[0.83,3.5],
    300: [0.83, 2.6],
   4: [1.2, 2.6],
    400: [1.2, 1.95],
    8: [1.79, 1.95],
    9: [2.5, 1.95],
    7: [1.79, 1],
    10: [2.5, 1],
    6: [1.2, 1],
    5: [0.8, 1],
    50: [0.8, 0.25],
    900: [3.2,1.95],
    24: [3.2,3.04],
    25: [4.75,3.04],
    26: [4.75,1.95],
    27: [5.22,1.95],
    28: [6.1,1.95],
    29: [6.1,1],
    30: [5.22,1],
    31: [4.32,1],
    32: [4.32,0.6],
    33: [3.5,0.6],
    310: [3.5,1]
        # Define node positions
        }

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(image, extent=[0, 8, 0, 4])
       
        
        nx.draw(graph, pos, with_labels=True, edge_color='black', node_color='lightgray',width = 0.01, node_size=250, ax=ax)

# Draw only the edges along the shortest path with red color
        edge_colors = ['#5bff03' if edge in shortest_path_graph.edges else 'black' for edge in graph.edges]
        
        nx.draw(shortest_path_graph, pos, with_labels=False, node_color='red', node_size=250, ax=ax)
        
        
        #source node icons
        #icon for source node
        source_node = n
        source_image = plt.imread('../capstone_flaskapp/static/source_node_icon.png')
        imagebox = OffsetImage(source_image, zoom=0.13)
        ab = AnnotationBbox(imagebox,(pos[source_node][0], pos[source_node][1]), frameon=False, boxcoords="data", pad=0)
        ax.add_artist(ab)
        #icon for destination node
        destination_node = 32  # Assuming 'm' is the destination node index
        destination_image = plt.imread('../capstone_flaskapp/static/destination_node_icon.png')
        imagebox_dest = OffsetImage(destination_image, zoom=0.03)
        ab_dest = AnnotationBbox(imagebox_dest, (pos[destination_node][0], pos[destination_node][1]+0.22), frameon=False, boxcoords="data", pad=0)
        ax.add_artist(ab_dest)

        # nx.draw_networkx_nodes(graph, pos, nodelist=[source_node], node_color='#f3f315', node_size=650, ax=ax)
        nx.draw_networkx_edges(graph, pos, edgelist=graph.edges, edge_color=edge_colors,width =0.01, ax=ax)
        nx.draw_networkx_edges(shortest_path_graph, pos, edgelist=shortest_path_graph.edges,edge_color='#5bff03', width=6.0, ax=ax)
        ax.set_xlim(0, 8)  
        ax.set_ylim(0, 4)  
        ax.axis('on')  
        # plt.grid(True) 
        # for x in range(9):
        #     for y in range(5):
        #         plt.text(x, y, f'({x:.1f}, {y:.1f})', ha='center', va='center', color='black', fontsize=8)
        img_path = '../capstone_flaskapp/static/uploads/shortest_path_image.png' 
        plt.savefig(img_path,bbox_inches='tight', pad_inches=1)
        plt.close()
        
        
        #13th floor
        
        shortest_path = nx.shortest_path(graph_2, source=19, target=selected_tag, weight='weight')
        
        i_path = '../capstone_flaskapp/static/blueprint_13.jpg' 
        image = plt.imread(i_path)
        shortest_path_graph = graph_2.subgraph(shortest_path)
        pos = {
            11:[5.5,3],
            110:[4.9,3],
            12:[3.2,3],
            120:[3.9,3],
            13:[4.4,3.5],
            130:[3.9,3.5],
            131:[4.9,3.5],
            14: [3.18,2.00],
            15: [5.4,1.00],
            16: [2.5,1.00],
            17: [3.35,1],
            170:[3.9,1],
            18:[3.9,2],
            180:[3.9,2.5],
            19:[4.5,0.3],
            190:[3.9,0.3],
            191:[4.9,0.3],
            20:[4.9,1],
            21:[4.4,2],
            22:[4.9,2],
            23:[5.45,2]
    
        # Define node positions
        }

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(image, extent=[0, 8, 0, 4])
        # nx.draw(graph, pos, with_labels=True, node_color='lightgray', node_size=150, ax=ax)
        
        nx.draw(graph_2, pos, with_labels=True, edge_color='black', node_color='lightgray',width = 0.01, node_size=250, ax=ax)

        #icon for source node
        source_node = 19
        source_image = plt.imread('../capstone_flaskapp/static/source_node_icon.png')
        imagebox = OffsetImage(source_image, zoom=0.13)
        ab = AnnotationBbox(imagebox,(pos[source_node][0], pos[source_node][1]), frameon=False, boxcoords="data", pad=0)
        ax.add_artist(ab)
        #icon for destination node
        destination_node = selected_tag  # Assuming 'm' is the destination node index
        destination_image = plt.imread('../capstone_flaskapp/static/destination_node_icon.png')
        imagebox_dest = OffsetImage(destination_image, zoom=0.03)
        ab_dest = AnnotationBbox(imagebox_dest, (pos[destination_node][0], pos[destination_node][1]+0.22), frameon=False, boxcoords="data", pad=0)
        ax.add_artist(ab_dest)

        # Draw only the edges along the shortest path with red color
        edge_colors = ['#5bff03' if edge in shortest_path_graph.edges else 'black' for edge in graph_2.edges]
        nx.draw(shortest_path_graph, pos, with_labels=False, node_color='red', node_size=250, ax=ax)
        nx.draw_networkx_edges(graph_2, pos, edgelist=graph_2.edges, edge_color=edge_colors,width =0.01, ax=ax)
        nx.draw_networkx_edges(shortest_path_graph, pos, edgelist=shortest_path_graph.edges,edge_color='#5bff03', width=6.0, ax=ax)
        ax.set_xlim(0, 8)  
        ax.set_ylim(0, 4)  
        ax.axis('on')  
        # plt.grid(True) 
        # for x in range(9):
        #     for y in range(5):
        #         plt.text(x, y, f'({x:.1f}, {y:.1f})', ha='center', va='center', color='black', fontsize=8)
       
        
        
        img_path_13 = '../capstone_flaskapp/static/uploads/blueprint_13.jpg' 
        plt.savefig(img_path_13,bbox_inches='tight', pad_inches=1)
        plt.close()  
        return jsonify({'image_url': '/static/uploads/shortest_path_image.png','image_url_13': '/static/uploads/blueprint_13.jpg','p1':"Proceed from your current position to Tag 32, and once there, take the elevator to the 13th floor. "}) 
    
    
    elif selected_tag is not None and (selected_tag<11 or selected_tag>23):
        shortest_path = nx.shortest_path(graph, source=n, target=selected_tag, weight='weight')
        i_path = '../capstone_flaskapp/static/shortest_path_image_1.png' 
        image = plt.imread(i_path)
        shortest_path_graph = graph.subgraph(shortest_path)
        pos = {
            1: [7.5,1.95],
            2: [7.5,1],
            3:[0.83,3.5],
    300: [0.83, 2.6],
   4: [1.2, 2.6],
    400: [1.2, 1.95],
    8: [1.79, 1.95],
    9: [2.5, 1.95],
    7: [1.79, 1],
    10: [2.5, 1],
    6: [1.2, 1],
    5: [0.8, 1],
    50: [0.8, 0.25],
    900: [3.2,1.95],
    24: [3.2,3.04],
    25: [4.75,3.04],
    26: [4.75,1.95],
    27: [5.22,1.95],
    28: [6.1,1.95],
    29: [6.1,1],
    30: [5.22,1],
    31: [4.32,1],
    32: [4.32,0.6],
    33: [3.5,0.6],
    310: [3.5,1]
        # Define node positions
        }

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(image, extent=[0, 8, 0, 4])
       
        
        nx.draw(graph, pos, with_labels=True, edge_color='black', node_color='lightgray',width = 0.01, node_size=250, ax=ax)

# Draw only the edges along the shortest path with red color
        edge_colors = ['#5bff03' if edge in shortest_path_graph.edges else 'black' for edge in graph.edges]
        
        nx.draw(shortest_path_graph, pos, with_labels=False, node_color='red', node_size=250, ax=ax)
        
        
        #source node icons
        source_node = n
        source_image = plt.imread('../capstone_flaskapp/static/source_node_icon.png')
        imagebox = OffsetImage(source_image, zoom=0.13)
        ab = AnnotationBbox(imagebox,(pos[source_node][0], pos[source_node][1]), frameon=False, boxcoords="data", pad=0)
        ax.add_artist(ab)
        #icon for destination node
        destination_node = selected_tag  # Assuming 'm' is the destination node index
        destination_image = plt.imread('../capstone_flaskapp/static/destination_node_icon.png')
        imagebox_dest = OffsetImage(destination_image, zoom=0.03)
        ab_dest = AnnotationBbox(imagebox_dest, (pos[destination_node][0], pos[destination_node][1]+0.22), frameon=False, boxcoords="data", pad=0)
        ax.add_artist(ab_dest)

        # nx.draw_networkx_nodes(graph, pos, nodelist=[source_node], node_color='#f3f315', node_size=650, ax=ax)
        nx.draw_networkx_edges(graph, pos, edgelist=graph.edges, edge_color=edge_colors,width =0.01, ax=ax)
        nx.draw_networkx_edges(shortest_path_graph, pos, edgelist=shortest_path_graph.edges,edge_color='#5bff03', width=6.0, ax=ax)
        ax.set_xlim(0, 8)  
        ax.set_ylim(0, 4)  
        ax.axis('on')  
        # plt.grid(True) 
        # for x in range(9):
        #     for y in range(5):
        #         plt.text(x, y, f'({x:.1f}, {y:.1f})', ha='center', va='center', color='black', fontsize=8)
       
        img_path = '../capstone_flaskapp/static/uploads/shortest_path_image.png' 
        plt.savefig(img_path)
        plt.close()  
        return jsonify({'image_url': '/static/uploads/shortest_path_image.png'})
    
    
    elif selected_tag>=11 and selected_tag<=23:
        shortest_path = nx.shortest_path(graph_2, source=n, target=selected_tag, weight='weight')
        
        i_path = '../capstone_flaskapp/static/blueprint_13.jpg' 
        image = plt.imread(i_path)
        shortest_path_graph = graph_2.subgraph(shortest_path)
        pos = {
            11:[5.5,3],
            110:[4.9,3],
            12:[3.2,3],
            120:[3.9,3],
            13:[4.4,3.5],
            130:[3.9,3.5],
            131:[4.9,3.5],
            14: [3.18,2.00],
            15: [5.4,1.00],
            16: [2.5,1.00],
            17: [3.35,1],
            170:[3.9,1],
            18:[3.9,2],
            180:[3.9,2.5],
            19:[4.5,0.3],
            190:[3.9,0.3],
            191:[4.9,0.3],
            20:[4.9,1],
            21:[4.4,2],
            22:[4.9,2],
            23:[5.45,2]
    
        # Define node positions
        }

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(image, extent=[0, 8, 0, 4])
        # nx.draw(graph, pos, with_labels=True, node_color='lightgray', node_size=150, ax=ax)
        
        nx.draw(graph_2, pos, with_labels=True, edge_color='black', node_color='lightgray',width = 0.01, node_size=250, ax=ax)

        #icon for source node
        source_node = n
        source_image = plt.imread('../capstone_flaskapp/static/source_node_icon.png')
        imagebox = OffsetImage(source_image, zoom=0.13)
        ab = AnnotationBbox(imagebox,(pos[source_node][0], pos[source_node][1]), frameon=False, boxcoords="data", pad=0)
        ax.add_artist(ab)
        #icon for destination node
        destination_node = selected_tag  # Assuming 'm' is the destination node index
        destination_image = plt.imread('../capstone_flaskapp/static/destination_node_icon.png')
        imagebox_dest = OffsetImage(destination_image, zoom=0.03)
        ab_dest = AnnotationBbox(imagebox_dest, (pos[destination_node][0], pos[destination_node][1]+0.22), frameon=False, boxcoords="data", pad=0)
        ax.add_artist(ab_dest)

        # Draw only the edges along the shortest path with red color
        edge_colors = ['#5bff03' if edge in shortest_path_graph.edges else 'black' for edge in graph_2.edges]
        nx.draw(shortest_path_graph, pos, with_labels=False, node_color='red', node_size=250, ax=ax)
        nx.draw_networkx_edges(graph_2, pos, edgelist=graph_2.edges, edge_color=edge_colors,width =0.01, ax=ax)
        nx.draw_networkx_edges(shortest_path_graph, pos, edgelist=shortest_path_graph.edges,edge_color='#5bff03', width=6.0, ax=ax)
        ax.set_xlim(0, 8)  
        ax.set_ylim(0, 4)  
        ax.axis('on')  
        # plt.grid(True) 
        # for x in range(9):
        #     for y in range(5):
        #         plt.text(x, y, f'({x:.1f}, {y:.1f})', ha='center', va='center', color='black', fontsize=8)
       
        img_path = '../capstone_flaskapp/static/uploads/blueprint_13.jpg' 
        plt.savefig(img_path)
        plt.close()  
        return jsonify({'image_url_13': '/static/uploads/blueprint_13.jpg','image_url': ''}) 
        
    else:
        return jsonify({'error': 'Selected tag is not set'})
@app.route('/public/<path:filename>')
def serve_public(filename):
    return send_from_directory('public', filename)

@app.route('/templates/<path:filename>')
def serve_template(filename):
    return send_from_directory('templates', filename)
    
'''
@app.route('/service-worker.js')
def service_worker():
    return send_from_directory('.', 'service-worker.js')
'''
if __name__ == '__main__':
    app.run(debug=True)
