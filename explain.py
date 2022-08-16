import json
import torch
from config import imgcap_gridTD_argument_parser
import torchvision.transforms as transforms
from dataset.dataloader import ImagecapDataset, ImagecapDatasetFromFeature
from models import gridTDmodel_lime
from models import explainerModelBU

import os
import yaml
import time
from datetime import datetime
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from sklearn.linear_model import Ridge
import glob
import math
import spacy
import nltk
import random
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

vg_classes = ['yolk', 'goal', 'bathroom', 'macaroni', 'umpire', 'toothpick', 'alarm clock', 'ceiling fan', 'photos', 'parrot', 'tail fin', 'birthday cake', 'calculator', 'catcher', 'toilet', 'batter', 'stop sign', 'cone', 'microwave', 'skateboard ramp', 'tea', 'dugout', 'products', 'halter', 'kettle', 'kitchen', 'refrigerator', 'ostrich', 'bathtub', 'blinds', 'court', 'urinal', 'knee pads', 'bed', 'flamingo', 'giraffe', 'helmet', 'giraffes', 'tennis court', 'motorcycle', 'laptop', 'tea pot', 'horse', 'television', 'shorts', 'manhole', 'dishwasher', 'jeans', 'sail', 'monitor', 'man', 'shirt', 'car', 'cat', 'garage door', 'bus', 'radiator', 'tights', 'sailboat', 'racket', 'plate', 'rock wall', 'beach', 'trolley', 'ocean', 'headboard', 'tea kettle', 'wetsuit', 'tennis racket', 'sink', 'train', 'keyboard', 'sky', 'match', 'train station', 'stereo', 'bats', 'tennis player', 'toilet brush', 'lighter', 'pepper shaker', 'gazebo', 'hair dryer', 'elephant', 'toilet seat', 'zebra', 'skateboard', 'zebras', 'floor lamp', 'french fries', 'woman', 'player', 'tower', 'bicycle', 'magazines', 'christmas tree', 'umbrella', 'cow', 'pants', 'bike', 'field', 'living room', 'latch', 'bedroom', 'grape', 'castle', 'table', 'swan', 'blender', 'orange', 'teddy bear', 'net', 'meter', 'baseball field', 'runway', 'screen', 'ski boot', 'dog', 'clock', 'hair', 'avocado', 'highway', 'skirt', 'frisbee', 'parasail', 'desk', 'pizza', 'mouse', 'sign', 'shower curtain', 'polar bear', 'airplane', 'jersey', 'reigns', 'hot dog', 'surfboard', 'couch', 'glass', 'snowboard', 'girl', 'plane', 'elephants', 'oven', 'dirt bike', 'tail wing', 'area rug', 'bear', 'washer', 'date', 'bow tie', 'cows', 'fire extinguisher', 'bamboo', 'wallet', 'tail feathers', 'truck', 'beach chair', 'boat', 'tablet', 'ceiling', 'chandelier', 'sheep', 'glasses', 'ram', 'kite', 'salad', 'pillow', 'fire hydrant', 'mug', 'tarmac', 'computer', 'swimsuit', 'tomato', 'tire', 'cauliflower', 'fireplace', 'snow', 'building', 'sandwich', 'weather vane', 'bird', 'jacket', 'chair', 'water', 'cats', 'soccer ball', 'horses', 'drapes', 'barn', 'engine', 'cake', 'head', 'head band', 'skier', 'town', 'bath tub', 'bowl', 'stove', 'tongue', 'coffee table', 'floor', 'uniform', 'ottoman', 'broccoli', 'olive', 'mound', 'pitcher', 'food', 'paintings', 'traffic light', 'parking meter', 'bananas', 'mountain', 'cage', 'hedge', 'motorcycles', 'wet suit', 'radish', 'teddy bears', 'monitors', 'suitcase', 'drawers', 'grass', 'apple', 'lamp', 'goggles', 'boy', 'armchair', 'ramp', 'burner', 'lamb', 'cup', 'tank top', 'boats', 'hat', 'soup', 'fence', 'necklace', 'visor', 'coffee', 'bottle', 'stool', 'shoe', 'surfer', 'stop', 'backpack', 'shin guard', 'wii remote', 'wall', 'pizza slice', 'home plate', 'van', 'packet', 'earrings', 'wristband', 'tracks', 'mitt', 'dome', 'snowboarder', 'faucet', 'toiletries', 'ski boots', 'room', 'fork', 'snow suit', 'banana slice', 'bench', 'tie', 'burners', 'stuffed animals', 'zoo', 'train platform', 'cupcake', 'curtain', 'ear', 'tissue box', 'bread', 'scissors', 'vase', 'herd', 'smoke', 'skylight', 'cub', 'tail', 'cutting board', 'wave', 'hedges', 'windshield', 'apples', 'mirror', 'license plate', 'tree', 'wheel', 'ski pole', 'clock tower', 'freezer', 'luggage', 'skateboarder', 'mousepad', 'road', 'bat', 'toilet tank', 'vanity', 'neck', 'cliff', 'tub', 'sprinkles', 'dresser', 'street', 'wing', 'suit', 'veggie', 'palm trees', 'urinals', 'door', 'propeller', 'keys', 'skate park', 'platform', 'pot', 'towel', 'computer monitor', 'flip flop', 'eggs', 'shed', 'moped', 'sand', 'face', 'scissor', 'carts', 'squash', 'pillows', 'family', 'glove', 'rug', 'watch', 'grafitti', 'dogs', 'scoreboard', 'basket', 'poster', 'duck', 'horns', 'bears', 'jeep', 'painting', 'lighthouse', 'remote control', 'toaster', 'vegetables', 'surfboards', 'ducks', 'lane', 'carrots', 'market', 'paper towels', 'island', 'blueberries', 'smile', 'balloons', 'stroller', 'napkin', 'towels', 'papers', 'person', 'train tracks', 'child', 'headband', 'pool', 'plant', 'harbor', 'counter', 'hand', 'house', 'donut', 'knot', 'soccer player', 'seagull', 'bottles', 'buses', 'coat', 'trees', 'geese', 'bun', 'toilet bowl', 'trunk', 'station', 'bikini', 'goatee', 'lounge chair', 'breakfast', 'nose', 'moon', 'river', 'racer', 'picture', 'shaker', 'sidewalk', 'shutters', 'stove top', 'church', 'lampshade', 'map', 'shop', 'platter', 'airport', 'hoodie', 'oranges', 'woods', 'enclosure', 'skatepark', 'vases', 'city', 'park', 'mailbox', 'balloon', 'billboard', 'pasture', 'portrait', 'forehead', 'ship', 'cookie', 'seaweed', 'sofa', 'slats', 'tomato slice', 'tractor', 'bull', 'suitcases', 'graffiti', 'policeman', 'remotes', 'pens', 'window sill', 'suspenders', 'easel', 'tray', 'straw', 'collar', 'shower', 'bag', 'scooter', 'tails', 'toilet lid', 'panda', 'comforter', 'outlet', 'stems', 'valley', 'flag', 'jockey', 'gravel', 'mouth', 'window', 'bridge', 'corn', 'mountains', 'beer', "pitcher's mound", 'palm tree', 'crowd', 'skis', 'phone', 'banana bunch', 'tennis shoe', 'ground', 'carpet', 'eye', 'urn', 'beak', 'giraffe head', 'steeple', 'mattress', 'baseball player', 'wine', 'water bottle', 'kitten', 'archway', 'candle', 'croissant', 'tennis ball', 'dress', 'column', 'utensils', 'cell phone', 'computer mouse', 'cap', 'lawn', 'airplanes', 'carriage', 'snout', 'cabinets', 'lemons', 'grill', 'umbrellas', 'meat', 'wagon', 'ipod', 'bookshelf', 'cart', 'roof', 'hay', 'ski pants', 'seat', 'mane', 'bikes', 'drawer', 'game', 'clock face', 'boys', 'rider', 'fire escape', 'slope', 'iphone', 'pumpkin', 'pan', 'chopsticks', 'hill', 'uniforms', 'cleat', 'costume', 'cabin', 'police officer', 'ears', 'egg', 'trash can', 'horn', 'arrow', 'toothbrush', 'carrot', 'banana', 'planes', 'garden', 'forest', 'brocolli', 'aircraft', 'front window', 'dashboard', 'statue', 'saucer', 'people', 'silverware', 'fruit', 'drain', 'jet', 'speaker', 'eyes', 'railway', 'lid', 'soap', 'rocks', 'office chair', 'door knob', 'banana peel', 'baseball game', 'asparagus', 'spoon', 'cabinet door', 'pineapple', 'traffic cone', 'nightstand', 'teapot', 'taxi', 'chimney', 'lake', 'suit jacket', 'train engine', 'ball', 'wrist band', 'pickle', 'fruits', 'pad', 'dispenser', 'bridle', 'breast', 'cones', 'headlight', 'necktie', 'skater', 'toilet paper', 'skyscraper', 'telephone', 'ox', 'roadway', 'sock', 'paddle', 'dishes', 'hills', 'street sign', 'headlights', 'benches', 'fuselage', 'card', 'napkins', 'bush', 'rice', 'computer screen', 'spokes', 'flowers', 'bucket', 'rock', 'pole', 'pear', 'sauce', 'store', 'juice', 'knobs', 'mustard', 'ski', 'stands', 'cabinet', 'dirt', 'goats', 'wine glass', 'spectators', 'crate', 'pancakes', 'kids', 'engines', 'shade', 'feeder', 'cellphone', 'pepper', 'blanket', 'sunglasses', 'train car', 'magnet', 'donuts', 'sweater', 'signal', 'advertisement', 'log', 'vent', 'whiskers', 'adult', 'arch', 'locomotive', 'tennis match', 'tent', 'motorbike', 'magnets', 'night', 'marina', 'wool', 'vest', 'railroad tracks', 'stuffed bear', 'moustache', 'bib', 'frame', 'snow pants', 'tank', 'undershirt', 'icons', 'neck tie', 'beams', 'baseball bat', 'safety cone', 'paper towel', 'bedspread', 'can', 'container', 'flower', 'vehicle', 'tomatoes', 'back wheel', 'soccer field', 'nostril', 'suv', 'buildings', 'canopy', 'flame', 'kid', 'baseball', 'throw pillow', 'belt', 'rainbow', 'lemon', 'oven door', 'tag', 'books', 'monument', 'men', 'shadow', 'bicycles', 'cars', 'lamp shade', 'pine tree', 'bouquet', 'toothpaste', 'potato', 'sinks', 'hook', 'switch', 'lamp post', 'lapel', 'desert', 'knob', 'chairs', 'pasta', 'feathers', 'hole', 'meal', 'station wagon', 'kites', 'boots', 'baby', 'biker', 'gate', 'signal light', 'headphones', 'goat', 'waves', 'bumper', 'bud', 'logo', 'curtains', 'american flag', 'yacht', 'box', 'baseball cap', 'fries', 'controller', 'awning', 'path', 'front legs', 'life jacket', 'purse', 'outfield', 'pigeon', 'toddler', 'beard', 'thumb', 'water tank', 'board', 'parade', 'robe', 'newspaper', 'wires', 'camera', 'pastries', 'deck', 'watermelon', 'clouds', 'deer', 'motorcyclist', 'kneepad', 'sneakers', 'women', 'onions', 'eyebrow', 'gas station', 'vane', 'girls', 'trash', 'numerals', 'knife', 'tags', 'light', 'bunch', 'outfit', 'groom', 'infield', 'frosting', 'forks', 'entertainment center', 'stuffed animal', 'yard', 'numeral', 'ladder', 'shoes', 'bracelet', 'teeth', 'guy', 'display case', 'cushion', 'post', 'pathway', 'tablecloth', 'skiers', 'trouser', 'cloud', 'hands', 'produce', 'beam', 'ketchup', 'paw', 'dish', 'raft', 'crosswalk', 'front wheel', 'toast', 'cattle', 'players', 'group', 'coffee pot', 'track', 'cowboy hat', 'petal', 'eyeglasses', 'handle', 'table cloth', 'jets', 'shakers', 'remote', 'snowsuit', 'bushes', 'dessert', 'leg', 'eagle', 'fire truck', 'game controller', 'smartphone', 'backsplash', 'trains', 'shore', 'signs', 'bell', 'cupboards', 'sweat band', 'sack', 'ankle', 'coin slot', 'bagel', 'masts', 'police', 'drawing', 'biscuit', 'toy', 'legs', 'pavement', 'outside', 'wheels', 'driver', 'numbers', 'blazer', 'pen', 'cabbage', 'trucks', 'key', 'saddle', 'pillow case', 'goose', 'label', 'boulder', 'pajamas', 'wrist', 'shelf', 'cross', 'coffee cup', 'foliage', 'lot', 'fry', 'air', 'officer', 'pepperoni', 'cheese', 'lady', 'kickstand', 'counter top', 'veggies', 'baseball uniform', 'book shelf', 'bags', 'pickles', 'stand', 'netting', 'lettuce', 'facial hair', 'lime', 'animals', 'drape', 'boot', 'railing', 'end table', 'shin guards', 'steps', 'trashcan', 'tusk', 'head light', 'walkway', 'cockpit', 'tennis net', 'animal', 'boardwalk', 'keypad', 'bookcase', 'blueberry', 'trash bag', 'ski poles', 'parking lot', 'gas tank', 'beds', 'fan', 'base', 'soap dispenser', 'banner', 'life vest', 'train front', 'word', 'cab', 'liquid', 'exhaust pipe', 'sneaker', 'light fixture', 'power lines', 'curb', 'scene', 'buttons', 'roman numerals', 'muzzle', 'sticker', 'bacon', 'pizzas', 'paper', 'feet', 'stairs', 'triangle', 'plants', 'rope', 'beans', 'brim', 'beverage', 'letters', 'soda', 'menu', 'finger', 'dvds', 'candles', 'picnic table', 'wine bottle', 'pencil', 'tree trunk', 'nail', 'mantle', 'countertop', 'view', 'line', 'motor bike', 'audience', 'traffic sign', 'arm', 'pedestrian', 'stabilizer', 'dock', 'doorway', 'bedding', 'end', 'worker', 'canal', 'crane', 'grate', 'little girl', 'rims', 'passenger car', 'plates', 'background', 'peel', 'brake light', 'roman numeral', 'string', 'tines', 'turf', 'armrest', 'shower head', 'leash', 'stones', 'stoplight', 'handle bars', 'front', 'scarf', 'band', 'jean', 'tennis', 'pile', 'doorknob', 'foot', 'houses', 'windows', 'restaurant', 'booth', 'cardboard box', 'fingers', 'mountain range', 'bleachers', 'rail', 'pastry', 'canoe', 'sun', 'eye glasses', 'salt shaker', 'number', 'fish', 'knee pad', 'fur', 'she', 'shower door', 'rod', 'branches', 'birds', 'printer', 'sunset', 'median', 'shutter', 'slice', 'heater', 'prongs', 'bathing suit', 'skiier', 'rack', 'book', 'blade', 'apartment', 'manhole cover', 'stools', 'overhang', 'door handle', 'couple', 'picture frame', 'chicken', 'planter', 'seats', 'hour hand', 'dvd player', 'ski slope', 'french fry', 'bowls', 'top', 'landing gear', 'coffee maker', 'melon', 'computers', 'light switch', 'jar', 'tv stand', 'overalls', 'garage', 'tabletop', 'writing', 'doors', 'stadium', 'placemat', 'air vent', 'trick', 'sled', 'mast', 'pond', 'steering wheel', 'baseball glove', 'watermark', 'pie', 'sandwhich', 'cpu', 'mushroom', 'power pole', 'dirt road', 'handles', 'speakers', 'fender', 'telephone pole', 'strawberry', 'mask', 'children', 'crust', 'art', 'rim', 'branch', 'display', 'grasses', 'photo', 'receipt', 'instructions', 'herbs', 'toys', 'handlebars', 'trailer', 'sandal', 'skull', 'hangar', 'pipe', 'office', 'chest', 'lamps', 'horizon', 'calendar', 'foam', 'stone', 'bars', 'button', 'poles', 'heart', 'hose', 'jet engine', 'potatoes', 'rain', 'magazine', 'chain', 'footboard', 'tee shirt', 'design', 'walls', 'copyright', 'pictures', 'pillar', 'drink', 'barrier', 'boxes', 'chocolate', 'chef', 'slot', 'sweatpants', 'face mask', 'icing', 'wipers', 'circle', 'bin', 'kitty', 'electronics', 'wild', 'tiles', 'steam', 'lettering', 'bathroom sink', 'laptop computer', 'cherry', 'spire', 'conductor', 'sheet', 'slab', 'windshield wipers', 'storefront', 'hill side', 'spatula', 'tail light', 'bean', 'wire', 'intersection', 'pier', 'snow board', 'trunks', 'website', 'bolt', 'kayak', 'nuts', 'holder', 'turbine', 'stop light', 'olives', 'ball cap', 'burger', 'barrel', 'fans', 'beanie', 'stem', 'lines', 'traffic signal', 'sweatshirt', 'handbag', 'mulch', 'socks', 'landscape', 'soda can', 'shelves', 'ski lift', 'cord', 'vegetable', 'apron', 'blind', 'bracelets', 'stickers', 'traffic', 'strip', 'tennis shoes', 'swim trunks', 'hillside', 'sandals', 'concrete', 'lips', 'butter knife', 'words', 'leaves', 'train cars', 'spoke', 'cereal', 'pine trees', 'cooler', 'bangs', 'half', 'sheets', 'figurine', 'park bench', 'stack', 'second floor', 'motor', 'hand towel', 'wristwatch', 'spectator', 'tissues', 'flip flops', 'quilt', 'floret', 'calf', 'back pack', 'grapes', 'ski tracks', 'skin', 'bow', 'controls', 'dinner', 'baseball players', 'ad', 'ribbon', 'hotel', 'sea', 'cover', 'tarp', 'weather', 'notebook', 'mustache', 'stone wall', 'closet', 'statues', 'bank', 'skateboards', 'butter', 'dress shirt', 'knee', 'wood', 'laptops', 'cuff', 'hubcap', 'wings', 'range', 'structure', 'balls', 'tunnel', 'globe', 'utensil', 'dumpster', 'cd', 'floors', 'wrapper', 'folder', 'pocket', 'mother', 'ski goggles', 'posts', 'power line', 'wake', 'roses', 'train track', 'reflection', 'air conditioner', 'referee', 'barricade', 'baseball mitt', 'mouse pad', 'garbage can', 'buckle', 'footprints', 'lights', 'muffin', 'bracket', 'plug', 'taxi cab', 'drinks', 'surfers', 'arrows', 'control panel', 'ring', 'twigs', 'soil', 'skies', 'clock hand', 'caboose', 'playground', 'mango', 'stump', 'brick wall', 'screw', 'minivan', 'leaf', 'fencing', 'ledge', 'clothes', 'grass field', 'plumbing', 'blouse', 'patch', 'scaffolding', 'hamburger', 'utility pole', 'teddy', 'rose', 'skillet', 'cycle', 'cable', 'gloves', 'bark', 'decoration', 'tables', 'palm', 'wii', 'mountain top', 'shrub', 'hoof', 'celery', 'beads', 'plaque', 'flooring', 'surf', 'cloth', 'passenger', 'spot', 'plastic', 'knives', 'case', 'railroad', 'pony', 'muffler', 'hot dogs', 'stripe', 'scale', 'block', 'recliner', 'body', 'shades', 'tap', 'tools', 'cupboard', 'wallpaper', 'sculpture', 'surface', 'sedan', 'distance', 'shrubs', 'skiis', 'lift', 'bottom', 'cleats', 'roll', 'clothing', 'bed frame', 'slacks', 'tail lights', 'doll', 'traffic lights', 'symbol', 'strings', 'fixtures', 'short', 'paint', 'candle holder', 'guard rail', 'cyclist', 'tree branches', 'ripples', 'gear', 'waist', 'trash bin', 'onion', 'home', 'side mirror', 'brush', 'sweatband', 'handlebar', 'light pole', 'street lamp', 'pads', 'ham', 'artwork', 'reflector', 'figure', 'tile', 'mountainside', 'black', 'bricks', 'paper plate', 'stick', 'beef', 'patio', 'weeds', 'back', 'sausage', 'paws', 'farm', 'decal', 'harness', 'monkey', 'fence post', 'door frame', 'stripes', 'clocks', 'ponytail', 'toppings', 'strap', 'carton', 'greens', 'chin', 'lunch', 'name', 'earring', 'area', 'tshirt', 'cream', 'rails', 'cushions', 'lanyard', 'brick', 'hallway', 'cucumber', 'wire fence', 'fern', 'tangerine', 'windowsill', 'pipes', 'package', 'wheelchair', 'chips', 'driveway', 'tattoo', 'side window', 'stairway', 'basin', 'machine', 'table lamp', 'radio', 'pony tail', 'ocean water', 'inside', 'cargo', 'overpass', 'mat', 'socket', 'flower pot', 'tree line', 'sign post', 'tube', 'dial', 'splash', 'male', 'lantern', 'lipstick', 'lip', 'tongs', 'ski suit', 'trail', 'passenger train', 'bandana', 'antelope', 'designs', 'tents', 'photograph', "catcher's mitt", 'electrical outlet', 'tires', 'boulders', 'mannequin', 'plain', 'layer', 'mushrooms', 'strawberries', 'piece', 'oar', 'bike rack', 'slices', 'arms', 'fin', 'shadows', 'hood', 'windshield wiper', 'letter', 'dot', 'bus stop', 'railings', 'pebbles', 'mud', 'claws', 'police car', 'crown', 'meters', 'name tag', 'entrance', 'staircase', 'shrimp', 'ladies', 'peak', 'vines', 'computer keyboard', 'glass door', 'pears', 'pant', 'wine glasses', 'stall', 'asphalt', 'columns', 'sleeve', 'pack', 'cheek', 'baskets', 'land', 'day', 'blocks', 'courtyard', 'pedal', 'panel', 'seeds', 'balcony', 'yellow', 'disc', 'young man', 'eyebrows', 'crumbs', 'spinach', 'emblem', 'object', 'bar', 'cardboard', 'tissue', 'light post', 'ski jacket', 'seasoning', 'parasol', 'terminal', 'surfing', 'streetlight', 'alley', 'cords', 'image', 'jug', 'antenna', 'puppy', 'berries', 'diamond', 'pans', 'fountain', 'foreground', 'syrup', 'bride', 'spray', 'license', 'peppers', 'passengers', 'cement', 'flags', 'shack', 'trough', 'objects', 'arches', 'streamer', 'pots', 'border', 'baseboard', 'beer bottle', 'wrist watch', 'tile floor', 'page', 'pin', 'items', 'baseline', 'hanger', 'tree branch', 'tusks', 'donkey', 'containers', 'condiments', 'device', 'envelope', 'parachute', 'mesh', 'hut', 'butterfly', 'salt', 'restroom', 'twig', 'pilot', 'ivy', 'furniture', 'clay', 'print', 'sandwiches', 'lion', 'shingles', 'pillars', 'vehicles', 'panes', 'shoreline', 'stream', 'control', 'lock', 'microphone', 'blades', 'towel rack', 'coaster', 'star', 'petals', 'text', 'feather', 'spots', 'buoy']

def main(args, perturb, explmethod, sd):

    word_map_path = f'./dataset/wordmap_{args.dataset}.json'
    word_map = json.load(open(word_map_path, 'r'))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])



    print('==========Loading Data==========')
    val_data = ImagecapDatasetFromFeature(args.dataset, args.test_split, val_transform, )
    
    if args.test_split == 'train':
        with open ('random_train_indexes', 'rb') as fp:
            subset_indices = pickle.load(fp)
        subset = torch.utils.data.Subset(val_data, subset_indices)
        val_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    else :
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    print(len(val_loader))




    print('==========Setting Model==========')
    model = gridTDmodel_lime.GridTDModelBU(args.embed_dim, args.hidden_dim, len(word_map), args.encoder)
    model.cuda()


    if args.weight:
        print(f'==========Resuming weights from {args.weight}==========')
        checkpoint = torch.load(args.weight)
        start_epoch = checkpoint['epoch']
        # epochs_since_improvement = checkpoint['epochs_since_improvement']
        # best_cider = checkpoint['cider']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print(f'==========Initializing model from random==========')
        start_epoch = 0
        epochs_since_improvement = 0
        best_cider = 0


    
    if explmethod == "none":
        prepare_data_lime(val_loader, model, word_map, args, perturb, explmethod, sd, pos=None, start_epoch=start_epoch)
        
    elif explmethod == "lime1":
        sd = "min"
        print(f"Mode : LIME1 (0-2 features perturb at once) | Std : {sd}")
        #pos = [None] + list(combinations(range(36), 1)) + list(combinations(range(36), 2)) + list(combinations(range(36), 3))
        pos = [None] + list(combinations(range(36), 1)) + list(combinations(range(36), 2))
        prepare_data_lime(val_loader, model, word_map, args, perturb, explmethod, sd, pos=pos, start_epoch=start_epoch)

    elif explmethod == "lime2":
        sd = "min"
        print(f"Mode : LIME2 (5 features perturb at once) | Std : {sd}")
        with open("lime2_perturb_indexes", 'rb') as f:
            while True:
                try:
                    pos = pickle.load(f)
                except EOFError:
                    f.close()
                    break
        prepare_data_lime(val_loader, model, word_map, args, perturb, explmethod, sd, pos=pos, start_epoch=start_epoch)

    elif explmethod == "lime3":
        sd = 1.5
        print(f"Mode : LIME3 (0-2 features perturb at once) | Std : {sd}")
        pos = [None] + list(combinations(range(36), 1)) + list(combinations(range(36), 2))
        prepare_data_lime(val_loader, model, word_map, args, perturb, explmethod, sd, pos=pos, start_epoch=start_epoch)
    
    elif explmethod == "lime4":
        sd = "min"
        print(f"Mode : LIME4 (13 features perturb at once) | Std : {sd}")
        with open("lime4_perturb_indexes", 'rb') as f:
            while True:
                try:
                    pos = pickle.load(f)
                except EOFError:
                    f.close()
                    break
        prepare_data_lime(val_loader, model, word_map, args, perturb, explmethod, sd, pos=pos, start_epoch=start_epoch)

    elif explmethod == "lime5":
        sd = "min"
        print(f"Mode : LIME5 (13 features perturb at once, 500 instances) | Std : {sd}")
        with open("lime5_perturb_indexes", 'rb') as f:
            while True:
                try:
                    pos = pickle.load(f)
                except EOFError:
                    f.close()
                    break
        prepare_data_lime(val_loader, model, word_map, args, perturb, explmethod, sd, pos=pos, start_epoch=start_epoch)

    elif explmethod == "lime_5obj" or explmethod == "lime_8obj": #there are two version, depending on the value of n below n=5, n=8
        sd = "min"
        n = int(explmethod[5])
        print(f"Mode : LIME obj ({n} objects) | Std : {sd}")
        targets = []
        for i in range(1, n+1):
            targets.append(list(combinations(range(n), i)))
        targets = [item for sublist in targets for item in sublist]
        prepare_data_lime(val_loader, model, word_map, args, perturb, explmethod, sd, pos=targets, start_epoch=start_epoch)  

    elif explmethod == "lrp":
        explainerModelBU.test_lrp_bu(args, word_map)

    else :
        print("Invalid perturbation type")
        
def prepare_data_lime(val_loader, model, word_map, args, perturb, explmethod, sd, start_epoch, beam_size=3, pos=-1, iteration=None):

    model.eval()
    rev_word_map = {v: k for k, v in word_map.items()}
    if explmethod=="lime1":
        perturbation = "_perturb_0_2_VF_stdmin"
    elif explmethod=="lime2":
        perturbation = "_perturb_5_VF_stdmin"
    elif explmethod=="lime3":
        perturbation = "_perturb_0_2_VF_std15"
    elif explmethod=="lime4":
        perturbation = "_perturb_13_VF_stdmin"
    elif explmethod=="lime5":
        perturbation = "_perturb_13_500_VF_stdmin"
    elif explmethod=="lime_5obj" or explmethod=="lime_8obj":
        perturbation = "_perturb_"+explmethod[5]+"_VFobj_stdmin"
        targets = pos
        filepath = f"./lime_probs/{args.dataset}/object_classes_scores_test"
        with open(filepath, 'rb') as f:
            while True:
                try:
                    all_obj_scores = pickle.load(f)
                except EOFError:
                    f.close()
                    break

    with torch.no_grad():
        prediction_save = {}
        image_id = 0        
        start_time = time.time()
        s = 0
        
        for i, (imgs, allcaps, caplens, img_filenames) in enumerate(val_loader):
            outprobs_list = []
            image_id += 1
            """
            if image_id <= 4000:
                continue
            """
            imgs = imgs.cuda()
            img_filename = img_filenames[0]
            iiiii = img_filename.split(".")[0]
            
            try:
                assert os.path.isfile(f"lime_probs/{args.dataset}/test/{iiiii}.jpg"+perturbation)
            except:
                if explmethod=="lime_5obj" or explmethod=="lime_8obj":
                    obj_scores = all_obj_scores[img_filename][0].detach().cpu().numpy()
                    unique_obj, unique_idx = np.unique(obj_scores, return_index=True)
                    if len(unique_obj) < int(explmethod[5]) :
                        continue
                    pos = [None]
                    for target in targets:
                        idxs = []
                        for item in target:
                            target_obj = unique_obj[item]
                            idxs_target_obj = np.where(obj_scores == target_obj)
                            idxs.append(idxs_target_obj[0].tolist())
                        idxs = [item for sublist in idxs for item in sublist]
                        pos.append(idxs)
                s += 1
                if s%100==0:
                    print(s, "perturbations done ...")
                    #continue
                for p in pos :
                    sentences, _ , outprobs = model.beam_search(imgs, word_map, perturb, p, sd, beam_size=beam_size)
                    outprobs_list.append(outprobs)

                    if img_filename not in prediction_save.keys(): #create new entry in prediction_save
                        prediction_save[img_filename] = []

                    for idx , sentence in enumerate(sentences):
                        prediction_save[img_filename].append(sentence)
                        #print("caption : ", sentence)

                prob_file_name = "lime_probs/"+args.dataset+"/"+args.test_split+"/"+img_filename+perturbation
                outprobs_list = torch.stack(outprobs_list)
                with open(prob_file_name, "wb") as result_file:
                    pickle.dump(outprobs_list, result_file)
                
        print("Finished ... total processed images : ", s)   
        print("Exec time : ", time.time()-start_time)

def ablation(args, perturb, explmethod, abltype, nb_abl, ablmagnitude, beam_size=3, iteration=None):


    if ablmagnitude == "15":
        ablmagnitude = 1.5

    nb_abl = int(nb_abl)

    word_map_path = f'./dataset/wordmap_{args.dataset}.json'
    word_map = json.load(open(word_map_path, 'r'))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])



    print('==========Loading Data==========')
    val_data = ImagecapDatasetFromFeature(args.dataset, args.test_split, val_transform, )
    
    if args.test_split == 'train':
        with open ('random_train_indexes', 'rb') as fp:
            subset_indices = pickle.load(fp)
        subset = torch.utils.data.Subset(val_data, subset_indices)
        val_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    else :
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    print('==========Setting Model==========')
    model = gridTDmodel_lime.GridTDModelBU(args.embed_dim, args.hidden_dim, len(word_map), args.encoder)
    model.cuda()

    if args.weight:
        print(f'==========Resuming weights from {args.weight}==========')
        checkpoint = torch.load(args.weight)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    #------------------------------------
    print(f"- {explmethod} : ")
    model.eval()
    nlp = spacy.load("en_core_web_sm")
    grammar = r"""
                    NP: {<NN><NN>} # Chunk two consecutive nouns
                    {<NNP><NNP>}  # Chunk two consecutive nouns phrases
    """
    cp = nltk.RegexpParser(grammar)
    rev_word_map = {v: k for k, v in word_map.items()}

    if explmethod in ["lime1", "lime2", "lime3", "lime4", "lime5", "lime_5obj", "lime_8obj"]:
        explanation_dir = "lime"
        explanation_file = "./"+explanation_dir+"_probs/"+args.dataset+"/explanations_"+args.test_split+"_"+explmethod
    elif explmethod == "lrp" or explmethod == "random" or explmethod == "random_obj":
        explanation_dir = "lrp"
        explanation_file = "./"+explanation_dir+"_probs/"+args.dataset+"/explanations_"+args.test_split+"_lrp"
    else:
        print("Unsupported explanation method !")
    #-----------------------------------------
    if True:#explmethod=="lime_5obj" or explmethod == "lime_8obj" or explmethod == "random_obj":
        """
        n_comb = 5
        perturbation = "_perturb_"+str(n_comb)+"_VFobj_stdmin"
        targets = []
        for i in range(1, n_comb+1):
            targets.append(list(combinations(range(n_comb), i)))
        targets = [item for sublist in targets for item in sublist]
        """
        filepath = f"./lime_probs/{args.dataset}/object_classes_scores_test"
        with open(filepath, 'rb') as f:
            while True:
                try:
                    all_obj_scores = pickle.load(f)
                except EOFError:
                    f.close()
                    break
    #-----------------------------------------
    with open(explanation_file, 'rb') as f:
        while True:
            try:
                explanations = pickle.load(f)
            except EOFError:
                f.close()
                break
    with torch.no_grad():
        image_id = 0
        nb_ignored_images = 0
        nb_existing_words = 0
        nb_missing_words = 0
        nb_words = 0
        total_avg_drop = 0
        nb_prob_drops = 0
        start_time = time.time()
        for i, (imgs, allcaps, caplens, img_filenames) in enumerate(val_loader):
            image_id += 1
            imgs = imgs.cuda()
            img_filename = img_filenames[0]
            print("---------------------------------------------------------- : ", img_filename)
            try :
                item = explanations[img_filename]
                if True:#explmethod == "lime_5obj" or explmethod == "lime_8obj" or explmethod == "random_obj":
                    obj_scores = all_obj_scores[img_filename][0].detach().cpu().numpy()
                    #print("Object detection (indexes) : ", obj_scores)
                    obj_labels = [vg_classes[idx] for idx in obj_scores]
                    print("Object detection (labels)  : ", obj_labels)
                    unique_obj, unique_idx = np.unique(obj_scores, return_index=True)

                for item in explanations[img_filename]:
                    for word in item:
                        nb_words += 1
                        perturb_idxs = []
                        #----------------------------------------------------------------------------
                        if abltype == "ob":
                            print("Ablation of objects...")
                            if explmethod == "random_obj":
                                obj_to_abl = random.sample(range(len(unique_obj)), nb_abl)
                                for i in obj_to_abl:
                                    features = np.where(obj_scores == unique_obj[i])[0].tolist()
                                    perturb_idxs.append(features)
                                perturb_idxs = [item for sublist in perturb_idxs for item in sublist]

                            else:
                                sorted_coef = item[word]
                                #print("Sorted VF (by importance) : ", sorted_coef)
                                i = 0
                                obj_to_abl = []

                                while len(obj_to_abl) < nb_abl:
                                    stop = False
                                    idx_obj = 0
                                    while not stop and idx_obj < len(unique_obj):
                                        features = np.where(obj_scores == unique_obj[idx_obj])[0].tolist()
                                        if sorted_coef[i] in features and idx_obj not in obj_to_abl:
                                            stop = True
                                            obj_to_abl.append(idx_obj)
                                        idx_obj += 1
                                    i += 1
                                
                                #get the feature indexes of the objects concerned by the ablation
                                for i in obj_to_abl:
                                    features = np.where(obj_scores == unique_obj[i])[0].tolist()
                                    perturb_idxs.append(features)
                                perturb_idxs = [item for sublist in perturb_idxs for item in sublist]
                            #print("Obj to abl : ", obj_to_abl)
                            #print("Obj label to abl : ", [obj_labels[idx] for idx in obj_to_abl])
                        #----------------------------------------------------------------------------
                        elif abltype == "vf":
                            print("Ablation of features...")
                            if explmethod == "random":
                                perturb_idxs = random.sample(range(36), nb_abl)
                            else:
                                sorted_coef = item[word]
                                perturb_idxs = sorted_coef[:nb_abl]
                            print("Ptb indexes : ", perturb_idxs)
                        #----------------------------------------------------------------------------
                        ptb_obj, ptb_idx = np.unique([obj_labels[idx] for idx in perturb_idxs], return_index=True)
                        print("word to explain : ", word)
                        print("Ptb objects  : ", ptb_obj)
                        sentences_noAbl, _ , outprobs_noAbl = model.beam_search(imgs, word_map, "", 0, 0, beam_size=beam_size)
                        sentences, _ , outprobs_withAbl = model.beam_search(imgs, word_map, perturb, perturb_idxs, ablmagnitude, beam_size=beam_size)
                        word_prob_noAbl = outprobs_noAbl[word_map[word]].detach().cpu().numpy()
                        word_prob_withAbl = outprobs_withAbl[word_map[word]].detach().cpu().numpy()
                        
                        total_avg_drop += word_prob_noAbl-word_prob_withAbl
                        if word_prob_withAbl < word_prob_noAbl :
                            nb_prob_drops += 1

                        for idx , sentence in enumerate(sentences):
                            if word in obj_list(sentence, nlp, cp) :
                                nb_existing_words += 1
                            else:
                                nb_missing_words += 1
                        print('------------------')

            except (KeyError, ValueError):
                nb_ignored_images += 1
            """
            if image_id == 3:
                break
            """
            #break
        
        print()
        print("Total Frequency probability drop in pred score = ", nb_prob_drops/nb_words)
        print("Total Avg probability drop in pred score = ", total_avg_drop/nb_words)
        print("Total number of words = ", nb_words)
        print("Number of founded words = ", nb_existing_words)
        print("Number of missing words = ", nb_missing_words)
        print("Percentage of missing words = ", nb_missing_words/nb_words)
        print("Number of ignored images = ", nb_ignored_images)
        
        print("________________________________________________________________________")

def get_classes_scores():
    
    # Load VG Classes
    data_path = './external/buAtt/demo/data/genome/1600-400-20'
    vg_classes = []
    with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            vg_classes.append(object.split(',')[0].lower().strip())
    MetadataCatalog.get("vg").thing_classes = vg_classes
    cfg = get_cfg()
    cfg.merge_from_file("./external/buAtt/configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml")
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # VG Weight
    cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
    predictor = DefaultPredictor(cfg)
    scores_list = {}
    i=0
    if args.dataset == "coco2017":
        images_path = "./dataset/coco/images/val2017/*"
    elif args.dataset == "flickr30k":
        images_path = "./dataset/flickr/images/test/*"
    for im_path in glob.iglob(images_path):
        raw_image = cv2.imread(im_path)
        with torch.no_grad():
            raw_height, raw_width = raw_image.shape[:2]
            image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": raw_height, "width": raw_width}]
            images = predictor.model.preprocess_image(inputs)
            features = predictor.model.backbone(images.tensor)
            proposals, _ = predictor.model.proposal_generator(images, features, None)
            proposal = proposals[0]
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in predictor.model.roi_heads.in_features]
            box_features = predictor.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
            pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
            outputs = FastRCNNOutputs(
                predictor.model.roi_heads.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                predictor.model.roi_heads.smooth_l1_beta,
            )
            probs = outputs.predict_probs()[0]
            boxes = outputs.predict_boxes()[0]

            for nms_thresh in np.arange(0.5, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes, probs, image.shape[1:], 
                    score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
                )
                if len(ids) == NUM_OBJECTS:
                    break
            scores = detector_postprocess(instances, raw_height, raw_width)
            roi_features = feature_pooled[ids].detach()
            img_filename = im_path.split("/")[-1]
            if img_filename not in scores_list.keys(): #create new entry in prediction_save
                scores_list[img_filename] = []
            #print([scores.pred_classes, scores.scores])
            scores_list[img_filename].append([scores.pred_classes, scores.scores])
            i+=1
            #break
            if i%500==0:
                print(i)
    classes_scores_file_name = f"lime_probs/{args.dataset}/object_classes_probs_test"
    with open(classes_scores_file_name, "wb") as result_file:
        pickle.dump(scores_list, result_file)

def similarity_euc_dist(ref, p1):

    """
    d1 = math.sqrt(pow(ref-p1, 2))
    d2 = math.sqrt(pow(ref-p2, 2))
    """
    d1 = ref-p1
    if d1>0:
        d1 = 0
    else:
        d1 = p1-ref
    s1 = 1/(1+d1)

    """
    d2 = ref-p2
    if d2>0:
        d2 = 0
    else:
        d2 = p2-ref
    """
    #s2 = 1/(1+d2)

    return d1, s1

def obj_list(cap, nlp, cp):
    doc = nlp(cap)
    token_list_cand = []
    obj_list_cand = []
    for token in doc :
        token_list_cand.append((token.text, token.tag_))
        #print(token.text, token.tag_)
    cand_parse_result = cp.parse(token_list_cand)
    for elem in cand_parse_result:
        if isinstance(elem, nltk.Tree):
            obj = ""
            space = ""
            for (text, tag) in elem:
                obj += space+text
                space = " "
            obj_list_cand.append(obj)
        else:
            if elem[1] == 'NNP' or elem[1] == 'NNS' or elem[1] == 'NN' or elem[1] == 'NNPS':
                obj_list_cand.append(elem[0])
    
    return obj_list_cand

def evaluate(args, vg_classes, explmethod):
    nlp = spacy.load("en_core_web_sm")
    grammar = r"""
                    NP: {<NN><NN>} # Chunk two consecutive nouns
                    {<NNP><NNP>}  # Chunk two consecutive nouns phrases
    """
    cp = nltk.RegexpParser(grammar)
    word_map_path = f'./dataset/wordmap_{args.dataset}.json'
    word_map = json.load(open(word_map_path, 'r'))
    lime_dir = f'./lime_probs/{args.dataset}/test/'
    lrp_dir = f'./lrp_probs/{args.dataset}/test/'
    s_lime_list = []
    s_lrp_list = []
    d_lime_list = []
    d_lrp_list = []
    p_lime_list = []
    p_lrp_list = []
    s_random_list = []
    d_random_list = []
    ref_list = []
    explanation_list_lime = {}
    explanation_list_lrp = {}

    clf = Ridge(alpha=1.0)
    nb_common_words = 0
    nb_missing_words = 0
    n = 0
    nb_ptb_instances = 0
    #-----------------------------
    if explmethod=="lime1":
        perturbation = "_perturb_0_2_VF_stdmin"
        pos = [None] + list(combinations(range(36), 1)) + list(combinations(range(36), 2))
        nb_ptb_instances = 667

        ptb_indexes = []
        ptb_indexes.append([0]*36)
        for i in range(1, len(pos)) :
            v = [0]*36
            for idx in pos[i]:
                v[idx] = 1
            ptb_indexes.append(v)
    #-----------------------------
    elif explmethod=="lime2":
        perturbation = "_perturb_5_VF_stdmin"
        with open("lime2_perturb_indexes", 'rb') as f:
            while True:
                try:
                    pos = pickle.load(f)
                except EOFError:
                    f.close()
                    break
        #pos = [None] + pos
        nb_ptb_instances = 50

        ptb_indexes = []
        #ptb_indexes.append([0]*36)
        for i in range(0, len(pos)) :
            v = [0]*36
            for idx in pos[i]:
                v[idx] = 1
            ptb_indexes.append(v)
    #-----------------------------
    elif explmethod=="lime3":
        perturbation = "_perturb_0_2_VF_std15"
        pos = [None] + list(combinations(range(36), 1)) + list(combinations(range(36), 2))
        nb_ptb_instances = 667

        ptb_indexes = []
        ptb_indexes.append([0]*36)
        for i in range(1, len(pos)) :
            v = [0]*36
            for idx in pos[i]:
                v[idx] = 1
            ptb_indexes.append(v)
    #-----------------------------
    elif explmethod=="lime4":
        perturbation = "_perturb_13_VF_stdmin"
        with open("lime4_perturb_indexes", 'rb') as f:
            while True:
                try:
                    pos = pickle.load(f)
                except EOFError:
                    f.close()
                    break
        nb_ptb_instances = 50
        ptb_indexes = []
        for i in range(0, len(pos)) :
            v = [0]*36
            for idx in pos[i]:
                v[idx] = 1
            ptb_indexes.append(v)
    #-----------------------------
    elif explmethod=="lime5":
        perturbation = "_perturb_13_500_VF_stdmin"
        with open("lime5_perturb_indexes", 'rb') as f:
            while True:
                try:
                    pos = pickle.load(f)
                except EOFError:
                    f.close()
                    break
        nb_ptb_instances = 500
        ptb_indexes = []
        for i in range(0, len(pos)) :
            v = [0]*36
            for idx in pos[i]:
                v[idx] = 1
            ptb_indexes.append(v)
    #-----------------------------
    elif explmethod=="lime_5obj" or explmethod=="lime_8obj":
        perturbation = "_perturb_"+explmethod[5]+"_VFobj_stdmin"
        n_comb = int(explmethod[5])
        nb_ptb_instances = 1 #the first instance : no alteration
        targets = []
        for i in range(1, n_comb+1):
            l = list(combinations(range(n_comb), i))
            targets.append(l)
            nb_ptb_instances += len(l)
        targets = [item for sublist in targets for item in sublist]

    #x_lime = ptb_indexes
    #-------------------------------------------------------------------------------
    with open(f"./lrp_probs/{args.dataset}/captions.yaml", 'r') as f:
            captions = yaml.safe_load(f)
            f.close()
    #-------------------------------------------------------------------------------
    filepath = f"./lime_probs/{args.dataset}/object_classes_scores_test"
    with open(filepath, 'rb') as f:
        while True:
            try:
                all_obj_scores = pickle.load(f)
            except EOFError:
                f.close()
                break
    
    for img_filename in all_obj_scores.keys():
        #print(img_filename)
        obj_scores = all_obj_scores[img_filename][0].detach().cpu().numpy()
        unique_obj, unique_idx = np.unique(obj_scores, return_index=True)

        if explmethod == "lime_5obj" or explmethod == "lime_8obj":
            if len(unique_obj) < n_comb :
                continue
            pos = [None]
            for target in targets:
                idxs = []
                for item in target:
                    target_obj = unique_obj[item]
                    idxs_target_obj = np.where(obj_scores == target_obj)
                    idxs.append(idxs_target_obj[0].tolist())
                idxs = [item for sublist in idxs for item in sublist]
                pos.append(idxs)
            ptb_indexes = []
            ptb_indexes.append([0]*36)
            for i in range(1, len(pos)) :
                v = [0]*36
                for idx in pos[i]:
                    v[idx] = 1
                ptb_indexes.append(v)
        #-------------------------------------------------------------------------------
        if explmethod[0:4] == "lime":
            try:
                with open(lime_dir+img_filename+perturbation, 'rb') as f:
                    while True:
                        try:
                            lime_probs = pickle.load(f)
                            #print(len(lime_probs))
                        except EOFError:
                            f.close()
                            break
            except :
                print("File not found in LIME dir !")
                continue
        #-------------------------------------------------------------------------------
        if explmethod == "lrp":
            try :
                with open(lrp_dir+img_filename+".hdf5_lrp_probs", 'rb') as f:
                    while True:
                        try:
                            lrp_probs = pickle.load(f)
                        except EOFError:
                            f.close()
                            break
            except FileNotFoundError:
                print("File not found in LRP dir !")
                continue
        #------------------------------------------------------------------------
        n += 1
        x_lime = ptb_indexes
        unique_obj = obj_scores[np.sort(unique_idx)]
        unique_idx = [np.where(obj_scores == u)[0][0] for u in unique_obj]
        cap = captions[img_filename+".hdf5"][0]
        obj_list_cand = obj_list(cap, nlp, cp)
        found_obj_frcnn_idx_list = []
        nb_selected_obj = 0
        for obj in obj_list_cand:
            split_obj = obj.split(" ")
            try :
                i = np.where(unique_obj == vg_classes.index(obj))[0]
            except ValueError:
                continue

            abandon = False
            if i.size != 0:
                found_obj_idx = unique_idx[i[0]]
                found_obj_frcnn_idx_list.append(found_obj_idx)
            #case of composed expressions
            elif len(split_obj)>1:
                #print(" décomposition ! ")
                idxs_composed_word = []
                for word in split_obj:
                    try :
                        i = np.where(unique_obj == vg_classes.index(obj))[0]
                    except ValueError:
                        abandon = True
                        break

                    if i.size == 0:
                        abandon = True
                        break
                    else:
                        idxs_composed_word.append(unique_idx[i[0]])
                if not abandon :
                    print("not abandon composed word !!")
                    found_obj_frcnn_idx_list.append(idxs_composed_word)
            else:
                #ni l'objet composé ni décomposé n'a été trouvé
                abandon = True
            if not abandon :
                ref = found_obj_frcnn_idx_list[nb_selected_obj]
                #----------------------------------------
                if explmethod == "lrp":
                    lrp_probs_obj = [0]*36
                    for w in split_obj:
                        lrp_probs_obj=np.add(lrp_probs_obj, lrp_probs[cap.split(" ").index(w)].detach().cpu().numpy())
                    lrp_probs_obj = lrp_probs_obj/len(split_obj)
                    sorted_coef_lrp =  np.argsort(-lrp_probs_obj)
                    p_lrp = np.where(sorted_coef_lrp == ref)[0][0]
                    ref_list.append(ref)
                    p_lrp_list.append(p_lrp)
                    d_lrp, s_lrp = similarity_euc_dist(ref, p_lrp)
                    s_lrp_list.append(s_lrp)
                    d_lrp_list.append(d_lrp)# ref-p_lrp
                    """
                    #To store the explanations later in a file
                    if img_filename not in explanation_list_lrp.keys():
                        explanation_list_lrp[img_filename] = []
                    explanation_list_lrp[img_filename].append({obj: sorted_coef_lrp.tolist()})
                    """
                #----------------------------------------
                elif explmethod[0:4] == "lime":
                    lime_probs_obj = [0]*nb_ptb_instances
                    for w in split_obj:
                        prob = lime_probs[:, word_map[w]]
                        lime_probs_obj = np.add(lime_probs_obj, prob.detach().cpu().numpy())
                    y_lime = lime_probs_obj/len(split_obj)
                    clf.fit(x_lime, y_lime)
                    sorted_coef_lime = np.argsort(-clf.coef_)
                    p_lime = np.where(sorted_coef_lime == ref)[0][0]
                    ref_list.append(ref)
                    p_lime_list.append(p_lime)
                    d_lime, s_lime = similarity_euc_dist(ref, p_lime)
                    s_lime_list.append(s_lime)
                    d_lime_list.append(d_lime)# ref-p_lime
                    """
                    if img_filename not in explanation_list_lime.keys():
                        explanation_list_lime[img_filename] = []
                    explanation_list_lime[img_filename].append({obj: sorted_coef_lime.tolist()})
                    """
                #----------------------------------------
                elif explmethod == "random":
                    coef_random = random.sample(range(36), 36)
                    p_random = np.where(coef_random == ref)[0][0]
                    d_random, s_random = similarity_euc_dist(ref, p_random)
                    s_random_list.append(s_random)
                    d_random_list.append(d_random) #ref-p_random
                    """
                    if img_filename not in explanation_list_random.keys():
                        explanation_list_random[img_filename] = []
                    explanation_list_random[img_filename].append({obj: coef_random.tolist()})
                    """
                #----------------------------------------
                nb_common_words +=1
                nb_selected_obj+=1

            else:
                nb_missing_words += 1
                continue
        """
        if n==30:
            break
        """
    print()
    print("-----------------------------")
    print(f"Results ({args.dataset}): ")
    print("Nb of common words = ", nb_common_words)
    print("Nb of non-common words = ", nb_missing_words)
    if explmethod == "lrp":
        if nb_common_words == 0 :
            score_lrp = 0
            avg_d_lrp = 0
        else:
            score_lrp = sum(s_lrp_list)/nb_common_words
            avg_d_lrp = sum(d_lrp_list)/nb_common_words
        print("Avg LRP distances  = ", avg_d_lrp)
        print("LRP score  = ", score_lrp)
    elif explmethod[0:4] == "lime":
        if nb_common_words == 0 :
            score_lime = 0
            avg_d_lime = 0
        else:
            score_lime = sum(s_lime_list)/nb_common_words
            avg_d_lime = sum(d_lime_list)/nb_common_words
        print("Avg LIME distances = ", avg_d_lime)
        print("LIME_score = ", score_lime)
    elif explmethod == "random":
        if nb_common_words == 0 :
            score_random = 0
            avg_d_random = 0
        else:
            score_random = sum(s_random_list)/nb_common_words
            avg_d_random = sum(d_random_list)/nb_common_words
        print("Avg RANDOM distances  = ", avg_d_random)
        print("RANDOM score  = ", score_random)

def generate_idxs(vg_classes):

    #unique_obj_nb_list = []
    idx_lists = {}

    n = 5
    targets = []
    for i in range(1, n+1):
        targets.append(list(combinations(range(n), i)))
    targets = [None] + [item for sublist in targets for item in sublist]

    filepath = f"./lime_probs/{args.dataset}/object_classes_scores_test"
    with open(filepath, 'rb') as f:
        while True:
            try:
                all_obj_scores = pickle.load(f)
            except EOFError:
                f.close()
                break

    for img_filename in all_obj_scores.keys():
        obj_scores = all_obj_scores[img_filename][0].detach().cpu().numpy()
        #t = [vg_classes[i] for i in obj_scores]
        #print(obj_scores)
        unique_obj, unique_idx = np.unique(obj_scores, return_index=True)
        #unique_obj_nb_list.append(len(unique_obj))
        #print(len(unique_obj))
        if len(unique_obj) > 4 :
            if img_filename not in idx_lists.keys(): #create new entry in prediction_save
                idx_lists[img_filename] = []
            idx_lists[img_filename].append([None, None]) #1st instance: no perturb
            for target in targets:
                #target_obj = unique_obj[random.randint(0,len(unique_obj)-1)] #randint:max bound included
                idxs_target_obj = np.where(obj_scores == target_obj)
                idx_lists[img_filename].append([target_obj, idxs_target_obj])
        #break
    #print(min(unique_obj_nb_list), max(unique_obj_nb_list))
    #count = len([i for i in unique_obj_nb_list if i > 9])
    #print(count)
    idxs_file_name = f"lime_probs/{args.dataset}/lime_5obj_idxs"
    with open(idxs_file_name, "wb") as result_file:
        pickle.dump(idx_lists, result_file)
    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    #print(model_weight_paths)
    parser = imgcap_gridTD_argument_parser()
    args = parser.parse_args()
    args.hidden_dim = 1024
    args.save_path = './output/gridTD_BU'

    if args.dataset == "coco2017":
        model_weight_paths= glob.glob('./output/gridTD_BU/vgg16/coco2017/checkpoint_coco2017_epoch21_cider_1.0862655929689613.pth')
    elif args.dataset == "flickr30k":
        model_weight_paths= glob.glob('./output/gridTD_BU/vgg16/flickr30k/checkpoint_flickr30k_epoch8_cider_0.5212386581763736.pth')
    else:
        print("Wrong dataset name... Supported datasets : coco2017, flickr30k")
        exit()
    args.weight = model_weight_paths[0]
    #args.dataset = 'coco2017'
    #args.dataset = 'flickr30k'
    
    supported_modes = ["none", "prepar", "expl", "eval"]
    supported_evals = ["correlation", "ablation"]
    supported_expl_models = ["random", "lrp", "lime1", "lime2", "lime3", "lime4", "lime5", "lime_5obj", "lime_8obj"]
    supported_abl_magnitudes = ["min", "max", "15"]
    supported_abl_nbonj = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    supported_abltypes = ["vf", "ob"]
    print("Dataset : ", args.dataset)
    print("Mode : ", args.mode)
    print("Explanation Model : ", args.expl_model)
    print('-------------------------')

    if args.mode == "prepar" :
        print(f"Preparing data for {args.expl_model} ...")
        print("Data already prepared !")
        #get_classes_scores()
        #generate_idxs(vg_classes)

    elif args.mode == "expl" and args.expl_model in supported_expl_models:
        main(args=args, perturb="VF", explmethod=args.expl_model, sd="")

    elif args.mode == "eval" and args.expl_model in supported_expl_models:
        if args.eval_type == "correlation":
            print("Correlation measure ... ")
            evaluate(args, vg_classes, explmethod=args.expl_model)
        elif args.eval_type in supported_evals and args.expl_model in supported_expl_models and args.abl_type[0:2] in supported_abltypes and args.abl_type[3:4] in supported_abl_nbonj and args.abl_type[5:] in supported_abl_magnitudes:
            print("Ablation study ... ")
            ablation(args=args, perturb="VF", explmethod=args.expl_model, abltype=args.abl_type[0:2], nb_abl=args.abl_type[3:4], ablmagnitude=args.abl_type[5:])
        else:
            print("Unsupported evaluation type !")
    elif args.mode == "none":
        main(args=args, perturb="VF", explmethod=args.mode, sd="")
    else :
        print("Unsupported mode or Explanation model ! \n --- Supported modes : prepar, expl, eval ---\n --- Supported expl models : lrp, lime1, lime2, lime3, lime4, lime5, lime_5obj, lime_8obj ---")

