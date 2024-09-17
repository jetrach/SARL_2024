%% Experiment 1 modeling code
%% 12/3/2020; New Haven, CT

clear all;close all;clc;

data = readtable('SARL_expt1_exclusionsn28_final.csv'); % load data

subs = unique(data.subject); % subject IDs
nsubs = length(subs); % N


for si = 1:nsubs

    subidx = strcmp(data.subject,subs{si}); % get subject si's ID
    subdata = data(subidx,:); % subset their data
    preRLidx = strcmp(subdata.rlType,'preRL'); % subset data of attention trials pre-RL trials

    RLidx = strcmp(subdata.trialType,'rl'); % RL trial indices
    rldata = subdata(RLidx,:); % subset RL trials


    bad =  strcmp(rldata.reward,'NA'); % timed out trials?
    reward = strcmp(rldata.reward,'1'); % reward outcome (1's and 0's)
    reward = double(reward); % format as vector
    reward(bad) = nan; % NaN the bad trials

    choiceShape = rldata.shapeChosen; % chosen shape stimulus
    nrl = length(choiceShape); % N rl trials?
    choice = nan(1,length(reward)); % init choice vector
    type = nan(1,nrl);cor = nan(1,nrl);rt = nan(1,nrl); % init some other stuff
    %% for loop for formatting the data in a way that is nice for modeling
    for t = 1:length(choiceShape)
        if strcmp(choiceShape{t},'circle')
            choice(t) = 1; % circle == choice 1
        elseif strcmp(choiceShape{t},'square')
            choice(t) = 2; % square == chocie 2
        else
            choice(t) = nan; % time out
        end
    end


    %%%%%%%%%%%%%%%%
    %% fit models %%
    %%%%%%%%%%%%%%%%
    n_fitting_iter = 200; % iterate fitting procedure with X differrent random start values to avoid local minima
    beta_prior = 1; % gamma prior on beta (temperature)

    disp(['now fitting subject ',num2str(si),' and model 2']); % print stuff to command window

    % fitting loop
    for k = 1:n_fitting_iter

        alpha = rand/10; % random learning starting final
        alphaneg = rand/10; % random negative learning starting final
        beta = rand*10; % random temperature

        params = [alpha,alphaneg,beta]; % parameters you are fitting
        options=optimset('display','off'); % some matlab stuff

        LB = [0 0 0]; % lower boundaries on the parameters, in the order seen in "params"
        UB = [1 1 50]; % upper boundaries on the parameters, in the order seen in "params"

        %% here is where matlab does its magic (fmincon)
        % the first entry of this function is key; it calls the function
        % script that actually represents the model you're fitting
        % last few inputs are actual subject data, and the "prior" thingy
        [params, ll] = fmincon(@func_2alpha,params,[],[],[],[],LB,UB,[],options,choice,reward,beta_prior);

        model1.p(k,:) = params; % store the params spit out as the best fit of this iteration
        model1.ll(k) = ll; % store the fit quality (negative log liklihood)
    end

    % now store the actual best fit over all the X iterations
    [modelFit_2a.ll(si),best] = min(model1.ll); % first find the BEST ONE (MINIMUM negative LL)
    modelFit_2a.alpha(si) = model1.p(best,1); % store best param X
    modelFit_2a.alphaneg(si) = model1.p(best,2);
    modelFit_2a.beta(si) = model1.p(best,3);
    modelFit_2a.n_params(si) = length(model1.p(1,:)); % store the number of free params in model
    % lastly, can compute and store aic and bic values for each subject's fit
    [modelFit_2a.aic(si),modelFit_2a.bic(si)] = aicbic(-modelFit_2a.ll(si),modelFit_2a.n_params(si),sum(~isnan(choice)));



    %% now do it all again with a second model
    disp(['now fitting subject ',num2str(si),' and model 1']);

    for k = 1:n_fitting_iter

        alpha = rand/10;
        beta = rand*10;

        params = [alpha,beta];
        options=optimset('display','off');
        LB = [0 0];
        UB = [1 50];
        [params, ll] = fmincon(@func_1alpha,params,[],[],[],[],LB,UB,[],options,choice,reward,beta_prior);

        model2.p(k,:) = params;
        model2.ll(k) = ll;
    end

    [modelFit_1a.ll(si),best] = min(model2.ll);
    modelFit_1a.alpha(si) = model2.p(best,1);
    modelFit_1a.beta(si) = model2.p(best,2);
    modelFit_1a.n_params(si) = length(model2.p(1,:));
    [modelFit_1a.aic(si),modelFit_1a.bic(si)] = aicbic(-modelFit_1a.ll(si),modelFit_1a.n_params(si),sum(~isnan(choice)));


end

save models/modelFit_2a_new modelFit_2a
save models/modelFit_1a_new modelFit_1a
