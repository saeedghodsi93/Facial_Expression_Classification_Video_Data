clc;
clear all;
close all;

dataset = 'LFPW'; % 'AFLW', 'AFW', ''HELEN', 'IBUG', 'LFPW'
current_folder = 0; % load 1 image from current folder or all files within inputdir
only_valid_files = 0; % copy all files or just valid files
datasetsdir = 'D:\Saeed\Uni\Master\1\Computer Vision in Smart Environments\Project\Phase 1\Dataset';

if(current_folder==1)
    load('image.mat','pt2d');
    fid = fopen('image.pts','wt');
    fprintf(fid,'version: 1\nn_points:  21\n{\n');
    if(size(pt2d,2)==68)
        temp_pt2d = zeros(2,21);
        temp_pt2d(1,1) = pt2d(1,18);
        temp_pt2d(2,1) = pt2d(2,18);
        temp_pt2d(1,2) = pt2d(1,20);
        temp_pt2d(2,2) = pt2d(2,20);
        temp_pt2d(1,3) = pt2d(1,22);
        temp_pt2d(2,3) = pt2d(2,22);
        temp_pt2d(1,4) = pt2d(1,23);
        temp_pt2d(2,4) = pt2d(2,23);
        temp_pt2d(1,5) = pt2d(1,25);
        temp_pt2d(2,5) = pt2d(2,25);
        temp_pt2d(1,6) = pt2d(1,27);
        temp_pt2d(2,6) = pt2d(2,27);
        temp_pt2d(1,7) = pt2d(1,37);
        temp_pt2d(2,7) = pt2d(2,37);
        temp_pt2d(1,8) = (pt2d(1,38)+pt2d(1,39)+pt2d(1,41)+pt2d(1,42))/4;
        temp_pt2d(2,8) = (pt2d(2,38)+pt2d(2,39)+pt2d(2,41)+pt2d(2,42))/4;
        temp_pt2d(1,9) = pt2d(1,40);
        temp_pt2d(2,9) = pt2d(2,40);
        temp_pt2d(1,10) = pt2d(1,43);
        temp_pt2d(2,10) = pt2d(2,43);
        temp_pt2d(1,11) = (pt2d(1,44)+pt2d(1,45)+pt2d(1,47)+pt2d(1,48))/4;
        temp_pt2d(2,11) = (pt2d(2,44)+pt2d(2,45)+pt2d(2,47)+pt2d(2,48))/4;
        temp_pt2d(1,12) = pt2d(1,46);
        temp_pt2d(2,12) = pt2d(2,46);
        temp_pt2d(1,13) = pt2d(1,3);
        temp_pt2d(2,13) = pt2d(2,3);
        temp_pt2d(1,14) = pt2d(1,32);
        temp_pt2d(2,14) = pt2d(2,32);
        temp_pt2d(1,15) = pt2d(1,31);
        temp_pt2d(2,15) = pt2d(2,31);
        temp_pt2d(1,16) = pt2d(1,36);
        temp_pt2d(2,16) = pt2d(2,36);
        temp_pt2d(1,17) = pt2d(1,15);
        temp_pt2d(2,17) = pt2d(2,15);
        temp_pt2d(1,18) = pt2d(1,61);
        temp_pt2d(2,18) = pt2d(2,61);
        temp_pt2d(1,19) = (pt2d(1,63)+pt2d(1,67))/2;
        temp_pt2d(2,19) = (pt2d(2,63)+pt2d(2,67))/2;
        temp_pt2d(1,20) = pt2d(1,65);
        temp_pt2d(2,20) = pt2d(2,65);
        temp_pt2d(1,21) = pt2d(1,9);
        temp_pt2d(2,21) = pt2d(2,9);
        pt2d = temp_pt2d;
    end
    for idx=1:21
        fprintf(fid,'%f %f\n',pt2d(1,idx),pt2d(2,idx));
    end
    fprintf(fid,'}');
    fclose(fid);
    
else
    inputdir = strcat(datasetsdir,'\facial_landmark_database\');
    outputdir = strcat(datasetsdir,'\PTS Format\');
    switch dataset
        case 'AFLW'
            inputdir = strcat(inputdir,'AFLW2000');
            outputdir = strcat(outputdir,'AFLW');
        case 'AFW'
            inputdir = strcat(inputdir,'dataset1\AFW');
            outputdir = strcat(outputdir,'AFW');
        case 'HELEN'
            inputdir = strcat(inputdir,'dataset1\HELEN');
            outputdir = strcat(outputdir,'HELEN');
        case 'IBUG'
            inputdir = strcat(inputdir,'dataset1\IBUG');
            outputdir = strcat(outputdir,'IBUG');
        case 'LFPW'
            inputdir = strcat(inputdir,'dataset1\LFPW');
            outputdir = strcat(outputdir,'LFPW');
    end
    if(only_valid_files==1)
        outputdir = strcat(outputdir,'_Valid');
    end
    
    infnames = dir(inputdir);
    for k=1:length(infnames)
        fname = infnames(k).name;
        if(strfind(fname,'.mat'))
            fname = fname(1:end-4);
            load(strcat(inputdir,'\',fname),'pt2d');
            
            if(~((only_valid_files==1)&&(any(-1.000000==reshape(pt2d,1,size(pt2d,1)*size(pt2d,2)))==1)))
                display(fname);
                
                inimgfname = strcat(inputdir,'\',fname,'.jpg');
                outimgfname = strcat(outputdir,'\',fname,'.jpg');
                copyfile(inimgfname,outimgfname);
                
                fname = strcat(outputdir,'\',fname,'.pts');
                fid = fopen(fname,'wt');
                fprintf(fid,'version: 1\nn_points:  21\n{\n');
                if(size(pt2d,2)==68)
                    temp_pt2d = zeros(2,21);
                    temp_pt2d(1,1) = pt2d(1,18);
                    temp_pt2d(2,1) = pt2d(2,18);
                    temp_pt2d(1,2) = pt2d(1,20);
                    temp_pt2d(2,2) = pt2d(2,20);
                    temp_pt2d(1,3) = pt2d(1,22);
                    temp_pt2d(2,3) = pt2d(2,22);
                    temp_pt2d(1,4) = pt2d(1,23);
                    temp_pt2d(2,4) = pt2d(2,23);
                    temp_pt2d(1,5) = pt2d(1,25);
                    temp_pt2d(2,5) = pt2d(2,25);
                    temp_pt2d(1,6) = pt2d(1,27);
                    temp_pt2d(2,6) = pt2d(2,27);
                    temp_pt2d(1,7) = pt2d(1,37);
                    temp_pt2d(2,7) = pt2d(2,37);
                    temp_pt2d(1,8) = (pt2d(1,38)+pt2d(1,39)+pt2d(1,41)+pt2d(1,42))/4;
                    temp_pt2d(2,8) = (pt2d(2,38)+pt2d(2,39)+pt2d(2,41)+pt2d(2,42))/4;
                    temp_pt2d(1,9) = pt2d(1,40);
                    temp_pt2d(2,9) = pt2d(2,40);
                    temp_pt2d(1,10) = pt2d(1,43);
                    temp_pt2d(2,10) = pt2d(2,43);
                    temp_pt2d(1,11) = (pt2d(1,44)+pt2d(1,45)+pt2d(1,47)+pt2d(1,48))/4;
                    temp_pt2d(2,11) = (pt2d(2,44)+pt2d(2,45)+pt2d(2,47)+pt2d(2,48))/4;
                    temp_pt2d(1,12) = pt2d(1,46);
                    temp_pt2d(2,12) = pt2d(2,46);
                    temp_pt2d(1,13) = pt2d(1,3);
                    temp_pt2d(2,13) = pt2d(2,3);
                    temp_pt2d(1,14) = pt2d(1,32);
                    temp_pt2d(2,14) = pt2d(2,32);
                    temp_pt2d(1,15) = pt2d(1,31);
                    temp_pt2d(2,15) = pt2d(2,31);
                    temp_pt2d(1,16) = pt2d(1,36);
                    temp_pt2d(2,16) = pt2d(2,36);
                    temp_pt2d(1,17) = pt2d(1,15);
                    temp_pt2d(2,17) = pt2d(2,15);
                    temp_pt2d(1,18) = pt2d(1,61);
                    temp_pt2d(2,18) = pt2d(2,61);
                    temp_pt2d(1,19) = (pt2d(1,63)+pt2d(1,67))/2;
                    temp_pt2d(2,19) = (pt2d(2,63)+pt2d(2,67))/2;
                    temp_pt2d(1,20) = pt2d(1,65);
                    temp_pt2d(2,20) = pt2d(2,65);
                    temp_pt2d(1,21) = pt2d(1,9);
                    temp_pt2d(2,21) = pt2d(2,9);
                    pt2d = temp_pt2d;
                end
                for idx=1:21
                    fprintf(fid,'%f %f\n',pt2d(1,idx),pt2d(2,idx));
                end
                fprintf(fid,'}');
                fclose(fid);
            end
        end
    end
    
end
