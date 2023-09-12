


#Pre Processing to Use weighted boxes fusion ensembling
def normalize_list(pass_list):
    pass_list=pass_list.asnumpy().tolist()
    all_values = [val for sublist in pass_list for val in sublist]
    min_val = min(all_values)
    max_val = max(all_values)

    # Normalize the elements within each sublist
    normalized_list = [[(x - min_val) / (max_val - min_val) for x in sublist] for sublist in pass_list]
    return normalized_list

