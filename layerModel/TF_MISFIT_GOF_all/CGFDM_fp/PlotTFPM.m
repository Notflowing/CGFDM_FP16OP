disp('---------------------------------------------------');
disp('       Matlab sript for plotting TFPM misfits       ');

% Plots time-frequency TFPM misfit for the one- or three-component signals
%(log frequency axis, misfits are scaled to the maximum from all three components)

% Input file name: 'MISFIT-GOF.DAT'
%                  'TFPM1.DAT' (for one-component signals)
%                  'TFPM1.DAT','TFPM2.DAT','TFPM3.DAT' (for three-component signals)

%  M. Kristekova, 2009

clear all;
cmap=load('jet_modn');         % colorscale

fid=fopen('MISFIT-GOF.DAT');  % reading of control data of the misfits&GOFs computation
MISFIT=fscanf(fid,'%g',inf);
fmin=log10( MISFIT(1) );% nebude mu ta pomlcka robit problem ako minus?
fmax=log10( MISFIT(2) );
NFREQ= MISFIT(3);
N= MISFIT(4);
dt= MISFIT(5);
nc= MISFIT(6);           % number of components
TFPMmax = MISFIT(7+4*nc+1);    % max value of TFPM misfits from all three components
TFPMmax = MISFIT(7+4*nc+2);    % max value of TFPM misfits from all three components
%...
fclose(fid);

col_max = (fix(TFPMmax*100.)+1.); % rounding to the nearest larger INT value when expressed in [%]
col_max_tic = abs((fix(TFPMmax*10.)-1)/10.); % rounding to the nearest larger INT value when expressed in [%]

% in case of locally normalized TFPM
% col_max should be computed later as max value of TFPM values for given component just after reading from file with TFPM results
df=(fmax-fmin)/(NFREQ-1);                                         

xmin=0.;       % beginning time (time for the first sample in data)
xmax=dt*(N-1); % ending time
ymin=fmin;     % lower frequency limit
ymax=fmax;     % upper frequency limit

% Time Tics
dx=[xmin:1:xmax+1];                         
dxLabel={[xmin:1:xmax+1]};
% Frequency Ticks
y_lin=cat(2,[0.1:0.1:0.9],[1:1:9],[10:5:50]);
dyLabel={'0.1';'';'';'0.4';'';'';'';'';'';'1';'2';'';'';'';'';'';'';'';'10';'';'';'';'50';};
dy=log10(y_lin);            % recalculating to log scale

for i=1:1:NFREQ;		    % frequency vector for plotting in TF plane
  freq(i)=ymin+(i-1)*df;
end
for i=1:1:N;                % time vector for plotting in TF plane
  time(i)=xmin+dt*(i-1);	
end

for k=1:1:nc
    f_name =['TFPM',num2str( k,'%01.0f'),'.DAT'];
    fid=fopen(f_name);       % reading from file with TFPM"k" results
	for i=1:1:NFREQ;		 % number "k" in the file name is the number of the component
	  a=fscanf(fid,'%g',[1 N]); 
	  tfa(i,:)=a;                
	end
	fclose(fid);

	figure;
	[C,h,cf]=contourf(time,freq,tfa * 100,[-col_max:col_max/20:col_max]);
	set(h,'EdgeColor','none', 'FaceColor', 'flat');
	colormap(cmap); 
	tfamax = max( max( abs( tfa * 100) ) );
%     caxis([-col_max col_max]);  % setting limits for colorbar
    caxis([-0.4 0.4]);  % setting limits for colorbar



	set(gca,"FontName","DejaVu Sans" );
    set(gca,'XTick',[dx]);
	set(gca,'XTickLabel',dxLabel);
	set(gca,'TickDir','out');
	set(gca,'YTick',[dy]);
	set(gca,'YTickLabel',dyLabel);
	xlim([xmin xmax]);          % setting limits for axes
	ylim([ymin ymax]);
	set(gca,'FontSize',10);
	% set(gca,'FontSize',12);
    % xlabel('time [s]','FontSize',12)
    % ylabel('frequency [Hz]','FontSize',12)

    if k == 1
        title( "CGFDM: TFEM(Vz)", "FontSize", 10, 'FontWeight', 'normal' );
    else
        title( "CGFDM: TFEM(Vz)", "FontSize", 10, 'FontWeight', 'normal' );
    end

	width= 500;
    height=250;
	set(gcf,'Position',[50,50,width,height]);
    %set(gca, 'Font', 'Arial')
    xlabel('t(s)','FontSize',10)
    ylabel('Frequency(Hz)','FontSize',10)

	cbar_axes = colorbar;
    cbar_axes.Ticks = [-0.8:0.2:0.8];
%     set(cbar_axes,'YTick',[-col_max_tic:col_max_tic/4.:col_max_tic]);
%     set(cbar_axes,'YTickLabel',[-col_max_tic*100.:col_max_tic*25.:col_max_tic*100.],'YTickMode','manual');
    % xlabel(cbar_axes,'[%]','FontSize',12);
	set(get(cbar_axes, "Title"), 'string', '%','FontSize',10 );
	% save figure to TIFF or png file, resolution 300 dpi 
	set(gcf, 'PaperUnits', 'inches');
	set(gcf, 'PaperPositionMode', 'manual');
	set(gcf, 'PaperPosition', [0.25 0.25 8.0 6.0]);  
% 	fig_name =['TFPM', num2str( k,'%01.0f'),'.tiff'];
    fig_name1 =['CGFDM_TFPM', num2str( k,'%01.0f'),'.pdf'];
%     print('-f','-dtiff','-r300',fig_name)
    % print('-f','-dpng','-r300',fig_name1)
	exportgraphics( gcf, fig_name1, 'ContentType', 'vector' );
end;
