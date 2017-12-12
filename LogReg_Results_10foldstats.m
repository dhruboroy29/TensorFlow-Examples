function [Median,IQR,OpPoint]=LogReg_Results_10foldstats(round_folder,madtype)

cd(round_folder);

[Orig_opt,MAD_opt] = LogReg_Optimize_Beta_10foldstats(round_folder,madtype);

Median={'Original_filter','MAD_filter'};
IQR={'Original_filter','MAD_filter'};
OpPoint={'Original_filter','MAD_filter'};

M_orig = table2struct(readtable('testing.csv','Delimiter',',','ReadVariableNames',false));
M_MAD = table2struct(readtable(strcat('testing_',madtype,'.csv'),'Delimiter',',','ReadVariableNames',false));

orig_array = [];
mad_array = [];

for i=1:length(M_orig)
    for j=1:length(Orig_opt)
        if strcmp(Orig_opt{j,1}, M_orig(i).Var1)==1 && Orig_opt{j,2}==M_orig(i).Var3 && Orig_opt{j,3}==M_orig(i).Var4
            orig_array = [orig_array, M_orig(i).Var5];
        end
    end
end

for i=1:length(M_MAD)
    for j=1:length(MAD_opt)
        if strcmp(MAD_opt{j,1}, M_MAD(i).Var1)==1 && MAD_opt{j,2}==M_MAD(i).Var3 && MAD_opt{j,3}==M_MAD(i).Var4
            mad_array = [mad_array, M_MAD(i).Var5];
        end
    end
end

Median=[Median;[{median(orig_array)}, {median(mad_array)}]];
IQR=[IQR;[{iqr(orig_array)}, {iqr(mad_array)}]];
OpPoint=[OpPoint;[{quantile(orig_array,0.25)}, {quantile(orig_array,0.25)}]];
