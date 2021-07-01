import sys,os
import gzip

def concat_lines(file_name):
    in_file = gzip.open(file_name, 'rb')
    str_list = []
    i_line = 0
    for line in in_file:
        i_line += 1
        if(i_line == 1):
	    continue
        tmp_str = line[:-1] + ' '
        str_list.append(tmp_str)
    grid_str = ''.join(str_list)
    in_file.close()
    return grid_str

def write_shape_grid(file_idx, grid_str):
    out_file = open('%s_0.grid'%(file_idx), 'w')
    for component in grid_str.split():
        out_file.write('%s '%(component))
    out_file.close()

def split_grid(file_idx, i_prop, grid_str):
    out_file_1 = open('%s_%i.pos.grid'%(file_idx, i_prop+1), 'w')
    out_file_2 = open('%s_%i.neg.grid'%(file_idx, i_prop+1), 'w')
    n_pos = 0
    n_neg = 0
    for component in grid_str.split():
        value = float(component)
        if(value == 0.0):
            out_file_1.write('%s '%(component))
            out_file_2.write('%s '%(component))
        elif(value > 0.0):
            n_pos += 1
            out_file_1.write('%f '%(value))
            out_file_2.write('0.0 ')
        elif(value < 0.0):
            n_neg += 1
            out_file_1.write('0.0 ')
            out_file_2.write('%f '%(-value))
    out_file_1.close()
    out_file_2.close()
    return n_pos, n_neg

def gen_shape_zer_m(file_idx):
    os.system('./gen_zernike %s_0.grid -c 0.5'%(file_idx))
    zer_m = []
    shape_file = open('%s_0.grid.inv'%(file_idx), 'r')
    for line in shape_file:
        if(line[0:3] == '121'):
            continue
        zer_m.append(float(line.split()[0]))
    shape_file.close()
    return zer_m

def gen_zer_m(file_idx, i_prop, n_pos, n_neg):
    if(n_pos > 0):    
        os.system('./gen_zernike %s_%i.pos.grid -c 0.5'%(file_idx, i_prop+1))
        pos_zerd = []
        pos_file = open('%s_%i.pos.grid.inv'%(file_idx, i_prop+1), 'r')
        for line in pos_file:
            if(line[0:3] == '121'):
                continue
            pos_zerd.append(float(line.split()[0]))
        pos_file.close()
    else:
        pos_zerd = [0.0 for i in range(121)]
    if(n_neg > 0):
        os.system('./gen_zernike %s_%i.neg.grid -c 0.5'%(file_idx, i_prop+1))
        neg_zerd = []
        neg_file = open('%s_%i.neg.grid.inv'%(file_idx, i_prop+1), 'r')
        for line in neg_file:
            if(line[0:3] == '121'):
                continue
            neg_zerd.append(float(line.split()[0]))
        neg_file.close()
    else:
        neg_zerd = [0.0 for i in range(121)]   
    zer_m = pos_zerd + neg_zerd
    return zer_m

def gen_zer_d(zer_m):
    val_sqr = [val ** 2 for val in zer_m]
    norm_sqr = 0.0
    for val in val_sqr:
        norm_sqr += val
    norm = norm_sqr ** 0.5
    zer_d = [val / norm for val in zer_m]
    return zer_d

def write_zer_d(zer_d, out_file_name):
    out_file = open(out_file_name, 'w')
    for val in zer_d:
        out_file.write('%7.5f\n'%(val))
    out_file.close()

def run_code(item):
    file_idx = item
    # for properties
    for i_prop in range(3):
        file_name = '%s_dim256_column%i_interp_shell.situs.gz'%(file_idx, i_prop)
        grid_str = concat_lines(file_name)
        n_pos, n_neg = split_grid(file_idx, i_prop, grid_str)
        zer_m = gen_zer_m(file_idx, i_prop, n_pos, n_neg)
        zer_d = gen_zer_d(zer_m)
        out_file_name = '%s_%i.zerd'%(file_idx, i_prop+1)
        write_zer_d(zer_d, out_file_name)
        os.system('rm %s_%i.pos.grid %s_%i.neg.grid %s_%i.pos.grid.inv %s_%i.neg.grid.inv'
                  %(file_idx, i_prop+1, file_idx, i_prop+1, file_idx, i_prop+1, file_idx, i_prop+1))
    # for shape
    file_name = '%s_dim256_shape_interp_shell.situs.gz'%(file_idx)
    grid_str = concat_lines(file_name)
    write_shape_grid(file_idx, grid_str)
    zer_m = gen_shape_zer_m(file_idx)
    zer_d = gen_zer_d(zer_m)
    out_file_name = '%s_0.zerd'%(file_idx)
    write_zer_d(zer_d, out_file_name)
    os.system('rm %s_0.grid %s_0.grid.inv'%(file_idx, file_idx))
    
offpath = sys.argv[1]
offlist = [y for y in os.listdir(offpath) if '.off' in y]

for item in offlist:
    run_code(item.split('.')[0])
