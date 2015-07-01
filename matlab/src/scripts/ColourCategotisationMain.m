%% Initialisation

clearvars;
close all;
clc;

docomparegt = true;

ColourSpace = 'lab';

GroundTruth = [];
ImageRGB = WcsChart();

%% Colour categorisation

PlotResults = 1;

BelongingImage = rgb2belonging(ImageRGB, ColourSpace, PlotResults, GroundTruth);

%% compare with gt
if docomparegt
  BerlinBelonging = WcsResults({'berlin'});
  SturgeBelonging = WcsResults({'sturges'});
  
  fprintf('Berlin:\n');
  PlotColourNamingDifferences(BelongingImage, BerlinBelonging);
  
  fprintf('Sturges:\n');
  PlotColourNamingDifferences(BelongingImage, SturgeBelonging);  
end
