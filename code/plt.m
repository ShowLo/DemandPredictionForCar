clear; close all; clc;

load('./data/demand_10.mat');

time = 0 : 1/144 : 21 - 1/144;

totalDemand_time = sum(demand, 2);
figure;
plot(time, totalDemand_time);
xlabel('时间');
ylabel('需求量');
title('总需求量随时间变化趋势');
set(gca, 'XLim', [-0.2, 21.1]);

day = 1 : 21;
totalRegionDemand_time = sum(reshape(totalDemand_time, 144, 21));
figure;
plot(day, totalRegionDemand_time);
xlabel('日期');
ylabel('需求量');
title('每日总需求量随日期变化趋势');
set(gca, 'xtick', 1 : 21);
set(gca, 'xLim', [0, 22]);

oneDay = 11;
totalRegionDemand_11 = totalDemand_time(144 * (oneDay - 1) + 1 : 144 * oneDay);
figure;
plot(0 : 1/6 : 24 - 1/6, totalRegionDemand_11);
xlabel('小时');
ylabel('需求量');
title('11日总需求量随时间变化趋势');
set(gca, 'xtick', 0 : 24);
set(gca, 'xLim', [0, 24]);

oneDay = 1;
totalRegionDemand_1 = totalDemand_time(144 * (oneDay - 1) + 1 : 144 * oneDay);
figure;
plot(0 : 1/6 : 24 - 1/6, totalRegionDemand_1);
xlabel('小时');
ylabel('需求量');
title('1日总需求量随时间变化趋势');
set(gca, 'xtick', 0 : 24);
set(gca, 'xLim', [0, 24]);

regionDemand = sum(demand);
figure;
bar(1:66, regionDemand);
set(gca, 'xtick', 1 : 66);
xlabel('区域ID');
ylabel('需求量');
title('不同区域的总需求量');