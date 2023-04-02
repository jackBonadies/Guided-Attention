import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

def showAttnPlot(folder):
    max_width = 20

    def populateSubGridWithSubPlots(valid_maps, key, outer_i):


        # Create the first set of subplots
        inner_grid_1 = gridspec.GridSpecFromSubplotSpec(int(len(valid_maps) / max_width) + 1, max_width, subplot_spec=outer_grid[outer_i], wspace=0.1, hspace=0.01)
        #fig.text(0.25, 1.0, 'Set 1', ha='center', fontsize=16)

        k=0
        for pairs in valid_maps:
            parts = pairs[0].split('_')
            token = parts[1]
            iter = None
            sub_iter = None
            for i in range(0, len(parts)):
                if parts[i] == "iter":
                    iter = parts[i+1]
                if parts[i] == "subiter":
                    sub_iter = parts[i+1].replace(".png","")
            ax = fig.add_subplot(inner_grid_1[k])
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_title(f'{key}', fontsize=6)
            else:
                ax.set_title(f'{iter}.{sub_iter}', fontsize=6)
            ax.imshow(pairs[1])
            k += 1


    maps = sorted(list(os.listdir(folder)))

    valid_maps = {}
    for file in maps:
        img = plt.imread(os.path.join(folder, file))
        #valid_maps. #get token
        
        if img.shape[0] != 16:
            continue
        token = file.split('_')[2]
        if token not in valid_maps:
            valid_maps[token] = []
        valid_maps[token].append((file, img))

    num_tokens = len(valid_maps.keys())

    fig = plt.figure(figsize=(20, 5))
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)

    outer_grid = gridspec.GridSpec(num_tokens, 1, hspace=0.3)


    outer_i = 0
    for key in valid_maps.keys():
        populateSubGridWithSubPlots(valid_maps[key], key, outer_i)
        outer_i += 1

    # Display the resulting figure
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Requires Folder Name")
        exit(-1)
    folder = sys.argv[1]
    showAttnPlot(folder)
    exit(0)