clc
clear

maindir  = '';
ref_image = '/MNI152_T1_1mm_brain.nii.gz';
subFiles = listFile(maindir);


for i = 3:length(subFiles)
    subdir = subFiles(i);
    paitent_dirs =  [maindir '/' subdir{1}];
	[cur_dir,image_name,~] = fileparts(paitent_dirs);
    paitent_dir = [paitent_dirs '/' image_name];
    T1_image =  [paitent_dirs  '/T1brain.nii.gz'];
    T1w2MNI_mat = [paitent_dirs  '/T1main.mat'];
    out_T1 = [paitent_dirs '/T1main.nii.gz'];
    cmd = ['flirt -ref ' ref_image ' -in ' T1_image  ' -omat ' T1w2MNI_mat  ' -out ' out_T1  ' -bins 256 -cost normcorr -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12  -interp trilinear ']; 
    system(cmd);
    
    mask =  [paitents_dir  'T1mask.nii.gz'];
    out_mask = [paitent_dirs '/main_mask.nii.gz'];
    cmd = ['flirt -ref ' ref_image ' -in ' mask  ' -applyxfm -init ' T1w2MNI_mat  ' -out ' out_mask  ' -paddingsize 0.0 -interp nearestneighbour ']; 
    system(cmd);
    
    
    refT1_image = [paitent_dirs '/T1main.nii.gz'];
    
    T1C_image =  [paitent_dir  '_t1ce.nii'];
    T1CMNI_mat = [paitent_dirs '/Cmain.mat'];
    out_C = [paitent_dirs '/Cmain.nii.gz'];
    cmd = ['flirt -ref ' refT1_image ' -in ' T1C_image ' -omat ' T1CMNI_mat  ' -out ' out_C  ' -bins 256 -cost normmi -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12  -interp trilinear ']; 
    system(cmd);
    
    T2_image =  [paitent_dir  '_t2.nii'];
    T1T2MNI_mat = [paitent_dirs '/T2main.mat'];
    out_T2 = [paitent_dirs '/T2main.nii.gz'];
    cmd = ['flirt -ref ' refT1_image ' -in ' T2_image  ' -omat ' T1T2MNI_mat  ' -out ' out_T2  ' -bins 256 -cost normmi -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12  -interp trilinear ']; 
    system(cmd);
	
end
