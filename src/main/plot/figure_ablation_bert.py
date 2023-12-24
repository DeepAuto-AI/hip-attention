import os
import cv2
import numpy as np

def import_bert_visulization(path):
    assert os.path.exists(path)
    img = cv2.imread(path)
    print(img.shape)
    H, W, C = img.shape
    N_LAYER = 12
    N_HEAD = 12
    N_SAMPLE = 4
    item_height = item_width = W // N_HEAD
    layer_height = H // N_LAYER
    layer_header_height = layer_height - item_height * N_SAMPLE
    layers = []
    for ilayer in range(N_LAYER):
        y_start = ilayer * (H // N_LAYER) + layer_header_height
        rows = []
        for isample in range(N_SAMPLE):
            row = []
            for ihead in range(N_HEAD):
                x = ihead * item_width
                y = y_start + isample * item_height
                row.append(img[y:y+item_height, x:x+item_width, :])
            rows.append(row)
        layers.append(rows)
    return layers

def import_opt(path):
    assert os.path.exists(path)
    img = cv2.imread(os.path.join(path, 'l0.png'))
    print(img.shape)
    _H, W, C = img.shape
    N_LAYER = 12
    N_HEAD = 12
    N_SAMPLE = 4
    item_height = item_width = W // N_HEAD
    layer_height = _H
    layer_header_height = layer_height - item_height * N_SAMPLE
    layers = []
    for ilayer in range(N_LAYER):
        img = cv2.imread(os.path.join(path, f'l{ilayer}.png'))
        print(img.shape)
        y_start = layer_header_height
        rows = []
        for isample in range(N_SAMPLE):
            row = []
            for ihead in range(N_HEAD):
                x = ihead * item_width
                y = y_start + isample * item_height
                row.append(img[y:y+item_height, x:x+item_width, :])
            rows.append(row)
        layers.append(rows)
    return layers

def render_layer(path_background, start_x, start_y, end_x, end_y, margin_x, margin_y, items):
    assert os.path.exists(path_background)
    img = cv2.imread(path_background)
    print(img.shape)
    H, W, C = img.shape
    N_ROWS = len(items)
    N_COLS = len(items[0])
    
    range_w = end_x - start_x
    range_h = end_y - start_y
    item_w = (range_w - margin_x * (N_COLS - 1)) / N_COLS
    item_h = (range_h - margin_y * (N_ROWS - 1)) / N_ROWS
    
    for i in range(N_ROWS):
        for j in range(N_COLS):
            x = int(round(j * (item_w + margin_x) + start_x))
            y = int(round(i * (item_h + margin_y) + start_y))
            w, h = int(round(item_w)), int(round(item_h))
            img[y:y+h, x:x+w, :] = cv2.resize(items[i][j], (w, h), interpolation=cv2.INTER_NEAREST)
    
    return img

def render(items):
    layers = []
    for items_layer in items:
        layer = render_layer("./plots/main/figure_visualization_bert/visualization.png", 325, 17, 2341, 691, 12, 14, items_layer)
        layers.append(layer)
    img = np.concatenate(layers, axis=0)
    return img

def process_opt(in_path, out_path):
    items = import_opt(in_path)
    img = render(items)
    cv2.imwrite(out_path, img)

def process_bert(in_path, out_path):
    items = import_bert_visulization(in_path)
    img = render(items)
    cv2.imwrite(out_path, img)

if __name__ == '__main__':
    process_opt('plots/main/figure_visualization_opt/wikitext2_0', 'plots/main/figure_visualization_opt/vis_rendered_wikitext2_0.png')
    process_opt('plots/main/figure_visualization_opt/wikitext2_1', 'plots/main/figure_visualization_opt/vis_rendered_wikitext2_1.png')
    process_opt('plots/main/figure_visualization_opt/wikitext2_2', 'plots/main/figure_visualization_opt/vis_rendered_wikitext2_2.png')
    process_opt('plots/main/figure_visualization_opt/wikitext2_3', 'plots/main/figure_visualization_opt/vis_rendered_wikitext2_3.png')
    process_bert('plots/main/figure_visualization_bert/3.png', 'plots/main/figure_visualization_bert/viz_rendered_3.png')
    process_bert('plots/main/figure_visualization_bert/5.png', 'plots/main/figure_visualization_bert/viz_rendered_5.png')
    process_bert('plots/main/figure_visualization_bert/7.png', 'plots/main/figure_visualization_bert/viz_rendered_7.png')
    process_bert('plots/main/figure_visualization_bert/8.png', 'plots/main/figure_visualization_bert/viz_rendered_8.png')