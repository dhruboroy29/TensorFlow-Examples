function [Result_orig,Result_MAD]=LogReg_Optimize_Beta_10foldstats(round_folder,madtype)

cd(round_folder);

Result_orig={};
Result_MAD={};
envs=table2struct(readtable('env_processing_order.csv','Delimiter',',','ReadVariableNames',false));

M_MAD = table2struct(readtable(strcat('training_',madtype,'.csv')),'Delimiter',',','ReadVariableNames',false);
M_orig = table2struct(readtable('training.csv'),'Delimiter',',','ReadVariableNames',false);

for i=1:length(envs)
    % Best params for MAD
    max_score_mad = 0;
    for k=1:length(M_MAD)
        if strcmp(M_MAD(k).Var1,envs(i).Var1)==1
            arr = [M_MAD(k).Var4 M_MAD(k).Var5 M_MAD(k).Var6 M_MAD(k).Var7 M_MAD(k).Var8 M_MAD(k).Var9 M_MAD(k).Var10 M_MAD(k).Var11 M_MAD(k).Var12 M_MAD(k).Var13];
            score=median(arr);
            if score > max_score_mad
                    max_score_mad = score;
                ResultArr={M_MAD(k).Var1, M_MAD(k).Var2, M_MAD(k).Var3, median(arr)};
            end
        end
    end
    Result_MAD=[Result_MAD;ResultArr];
    
    % Best params for original
    max_score_orig = 0
    for k=1:length(M_orig)
        if strcmp(M_orig(k).Var1,envs(i).Var1)==1
            arr = [M_orig(k).Var4 M_orig(k).Var5 M_orig(k).Var6 M_orig(k).Var7 M_orig(k).Var8 M_orig(k).Var9 M_orig(k).Var10 M_orig(k).Var11 M_orig(k).Var12 M_orig(k).Var13];
            score=median(arr);
            if score > max_score_orig
                    max_score_orig = score;
                ResultArr={M_orig(k).Var1, M_orig(k).Var2, M_orig(k).Var3, median(arr)};
            end
        end
    end
    Result_orig=[Result_orig;ResultArr];
end
        
        