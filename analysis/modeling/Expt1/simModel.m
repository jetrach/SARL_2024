%% code for quick pass analysis on juliana's sustained attention RL data
%% 12/3/2020; New Haven, CT

clear all;close all;clc;


data = readtable('SARL_expt1_exclusionsn28_final.csv');

subs = unique(data.subject);
nsubs = length(subs);
load models/modelFit_2a
mod = modelFit_2a;

mean_sim_r = nan(nsubs,100);
mean_sim_cor = nan(nsubs,100);
mean_sub_r = nan(nsubs,100);
mean_sub_cor = nan(nsubs,100);

for si = 1:nsubs

    subidx = strcmp(data.subject,subs{si});
    subdata = data(subidx,:);
    % all SA trials
    attallidx = ~strcmp(subdata.rlType,'rl');
    attalldata = subdata(:,:);

    RLidx = strcmp(subdata.trialType,'rl');
    rldata = subdata(RLidx,:);
    bad =  strcmp(rldata.reward,'NA');
    reward = strcmp(rldata.reward,'1'); % reward outcome
    reward = double(reward);
    reward(bad) = nan;

    choiceShape = rldata.shapeChosen;
    nrl = length(choiceShape);
    choice = nan(1,length(reward));
    bandit_probs = nan(2,length(reward));
    type = nan(1,nrl);cor = nan(1,nrl);rt = nan(1,nrl);


    ntrials = length(choice);sub_cor=[];

    bandit_probs(1,:) = str2double(rldata.circleVal);
    bandit_probs(2,:) = str2double(rldata.squareVal);

    for t = 1:ntrials
        if strcmp(choiceShape{t},'circle')
            choice(t) = 1;
            sub_cor(t) = round(bandit_probs(1,t));
        elseif strcmp(choiceShape{t},'square')
            choice(t) = 2;
            sub_cor(t) = round(bandit_probs(2,t));
        else
            choice(t) = nan;
            sub_cor(t) = nan;
        end
    end
    % model
    alpha = mod.alpha(si);
    alphaneg = mod.alphaneg(si);
    beta = mod.beta(si);

    na = 2;

    nsims = 100;
    sim_r = nan(nsims,ntrials);

    for sim = 1:nsims

        q = ones(na,1)*(1/na);
        pa = zeros(1,2)';

        for n = 1:ntrials

            % choose bandit
            pol = (exp(q.*beta)./sum(exp(q.*beta)))';

            % % Now select a machine
            x = rand;
            counts = histc(x,[0,cumsum(pol)]); % Setting up bins of prob intervals, which one is rand number "x" in?
            a = find(counts==1); % action=find bins that contain x

            % was agent rewarded?
            x = rand;
            if x < bandit_probs(a,n)
                sim_r(sim,n) = 1;
            else
                sim_r(sim,n) = 0;
            end
            % was agent correct?
            if bandit_probs(a,n) > bandit_probs(3-a,n)
                sim_cor(sim,n) = 1;
            else
                sim_cor(sim,n) = 0;
            end

            if sim_r(sim,n) == 0
                lr = alphaneg;
            else
                lr = alpha;
            end

            %% Q-learning
            q(a) = q(a) + lr*(sim_r(sim,n)-q(a));


            %         % update previous action vector for stickyness
            pa(:) = 0;
            pa(a) = 1;

        end
    end
    mean_sim_r(si,1:ntrials) = nanmean(sim_r);
    mean_sim_cor(si,1:ntrials) = nanmean(sim_cor(:,1:ntrials));
    mean_sub_r(si,1:ntrials) = reward;
    mean_sub_cor(si,1:ntrials) = sub_cor;
end


figure;
plot(nanmean(mean_sub_cor),'-ok','MarkerFaceColor','k','linewidth',2,'markersize',5);hold on;
plot(nanmean(mean_sim_cor),'color',[.8 .3 .1],'linewidth',2);
plot([25,25],[0,1],'--','color',[.1 .4 .7]);
plot([50,50],[0,1],'--','color',[.1 .4 .7]);
plot([75,75],[0,1],'--','color',[.1 .4 .7]);
plot([1 100],[.5 .5],'color',[.6 .6 .6]);
xlabel('trial');ylabel('p(correct)');
legend('data','model','reversal','location','southwest');
box off;
set(gca,'tickdir','out','linewidth',2)
set(gcf,'position',[5 613 474 179]);
print -dtiff -r300 SARL_ntb_lab_model

figure;
plot(nanmean(mean_sub_r),'-ok');hold on;
plot(nanmean(mean_sim_r),'r');
title('rewards');
xlabel('trial');ylabel('p(reward)');

figure;
plot(nanmean(mean_sub_cor),'-ok','MarkerFaceColor','k','linewidth',2,'markersize',5);hold on;
plot(mean_sim_cor([6,12,18,20,23],:)','color',[.5 .5 .5],'linewidth',2);
plot(nanmean(mean_sim_cor),'color',[.8 .3 .1],'linewidth',2);
plot([25,25],[0,1],'--','color',[.1 .4 .7]);
plot([50,50],[0,1],'--','color',[.1 .4 .7]);
plot([75,75],[0,1],'--','color',[.1 .4 .7]);
plot([1 100],[.5 .5],'color',[.6 .6 .6]);
xlabel('trial');ylabel('p(correct)');
%legend('data','model','reversal','location','southwest');
box off;
set(gca,'tickdir','out','linewidth',2)
set(gcf,'position',[5 613 474 179]);
print -dtiff -r300 SARL_ntb_lab_model

% Assuming there are 28 subplots, and we want a 4x4 grid, you can use a 7x4 grid.
for sub = 1:28
    subplot(7,4, sub); % Create a subplot in a 7x4 grid
    plot(nanmean(mean_sub_cor),'MarkerFaceColor','k','linewidth',2,'markersize',5); hold on;
    plot(mean_sim_cor(sub,:)','color',[.8 .3 .1],'linewidth',2);
    plot([25,25],[0,1],'--','color',[.1 .4 .7]);
    plot([50,50],[0,1],'--','color',[.1 .4 .7]);
    plot([75,75],[0,1],'--','color',[.1 .4 .7]);
    plot([1 100],[.5 .5],'color',[.6 .6 .6]);
    xlabel('trial'); ylabel('p(correct)');
    %legend('data','model','reversal','location','southwest');
    title(sub)
    box off;
    set(gca,'tickdir','out','linewidth',2)
end
% Adjust the figure size if needed
set(gcf,'position',[5 613 1200 800]); % Adjust the figure size to accommodate the subplots






