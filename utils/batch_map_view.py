import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

def showAttnPlot(folder, only_token = None, view_losses = False):
    max_width = 20

    def populateSubGridWithSubPlots(valid_maps, key, outer_i, losses_for_token):


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

            append_to_title = ""
            if losses_for_token is not None and len(losses_for_token) > 0:
                _key = f'{iter}.{sub_iter}'
                append_to_title = f' l:{losses_for_token[_key]:.2f}'
            if k == 0:
                ax.set_title(f'{key}{append_to_title}', fontsize=6)
            else:
                ax.set_title(f'{iter}.{sub_iter}{append_to_title}', fontsize=6)
            ax.imshow(pairs[1])
            k += 1


    maps = sorted(list(os.listdir(folder)))

    losses_for_token = {} #key = iter.sub_iter
    log_file_name = folder + ".txt"
    if view_losses and os.path.exists(log_file_name):
        f = open(log_file_name, "r")
        for line in f.readlines():
            if f"loss for {only_token}" in line:
                _key = line.split(' ')[0]
                loss_token = float(line.split(':')[-1])
                losses_for_token[_key] = loss_token

    valid_maps = {}
    for file in maps:
        img = plt.imread(os.path.join(folder, file))
        #valid_maps. #get token
        
        if img.shape[0] != 16:
            continue
        token = file.split('_')[2]
        include_token = True
        if only_token is not None and only_token != token:
            include_token = False
        if include_token:
            if token not in valid_maps:
                valid_maps[token] = []
            valid_maps[token].append((file, img))

    num_tokens = len(valid_maps.keys())

    fig = plt.figure(figsize=(20, 5))
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)

    outer_grid = gridspec.GridSpec(num_tokens, 1, hspace=0.3)


    outer_i = 0
    for key in valid_maps.keys():
        populateSubGridWithSubPlots(valid_maps[key], key, outer_i, losses_for_token)
        outer_i += 1

    # Display the resulting figure
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Requires Folder Name")
        exit(-1)
    folder = sys.argv[1]
    token = sys.argv[2] if len(sys.argv) > 2 else None
    view_losses = sys.argv[3] if len(sys.argv) > 3 else False
    showAttnPlot(folder, token, view_losses)
    exit(0)