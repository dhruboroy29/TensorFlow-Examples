function LogReg_InfoGainMAD_Preprocess(base_path, round)

files = dir(strcat(base_path,'/Round', num2str(round)));
dirmask = [files.isdir];
dirs = files(dirmask);

subfolders=[];
for k=1:length(dirs)
    if isempty(strfind(dirs(k).name,'.'))
        ComputeInfoGainRelevanceMap_MAD(strcat(base_path,'/Round', num2str(round),'/',dirs(k).name));
    end
end