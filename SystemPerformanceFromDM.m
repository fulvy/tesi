function [AFF, RR, FAR, FRR, EER, GARZeroFAR, AUC, CMS, areaGI, Genuins, Impostors, Identified]  = SystemPerformanceFromDM(DM, Probe, Gallery, SRR, srr_th, titolo)

DM = (DM-min(DM(:)))/(max(DM(:))-min(DM(:)));

NUM_GALLERY_SUB = 1; %numero di foto per soggetto

probe_dim   = size(DM, 1);
gallery_dim = size(DM, 2);
mindim = min([probe_dim gallery_dim]);
%

if DM == 0
    RR = -1;
    FAR = -1;
    FRR = -1;
    EER = -1;
    CMS = zeros(1,100)-1;
    areaGI = -1;
    
    'Errore'
    
    return;
end

pos = find(SRR < srr_th);
DM(pos, 1) = -1;


% Calcoliamo FAR e FRR facendo variare la soglia th
x = 0:0.005:1;
FAR = zeros(1, size(x,2));
FRR = zeros(1, size(x,2));
cnt = 1;

AFF = 0; %risposte ritenute affidabili


PFRR = 0;
PFAR = 0;
for i=1:probe_dim
    if(DM(i,1) ~= -1)
        for j=1:gallery_dim
            if(Probe(i)==Gallery(j))
                PFRR = PFRR + 1;
            else
                PFAR = PFAR + 1;
            end
        end
    end
end

for th=0:0.005:1
    
    confronti_ok = 0;
    for i=1:probe_dim
        
        if(DM(i,1) ~= -1)
            
            for j=1:gallery_dim

                    % verifichiamo se si tratta di un FAR
                    if (DM(i, j) <= th) && (Probe(i)~=Gallery(j))
                        FAR(cnt) = FAR(cnt) + 1;
                    end

                    % verifichiamo se si tratta di un FRR
                    if (DM(i, j) >= th) && (Probe(i)==Gallery(j))
                        FRR(cnt) = FRR(cnt) + 1;                
                    end

            end
          
        end
                 
    end
   
    FAR(cnt) = FAR(cnt)./PFAR;
    FRR(cnt) = FRR(cnt)./PFRR;
    cnt = cnt + 1;
      
end

% if(tipo==1)
%     FAR = 0.8*FAR;
%     FRR = 0.8*FRR;
%     p = find(FAR>0.8);
%     FAR(p) = (1/0.8)*FAR(p);
%     p = find(FRR>0.8);
%     FRR(p) = (1/0.8)*FRR(p);
% end

% calcoliamo l'Equal Error Rate
EER = min(abs(FAR-FRR));
xEER = x(min(find(abs(FAR-FRR) == EER)));
EER = 0.5*(FAR(min(find(abs(FAR-FRR) == EER)))  +  FRR(min(find(abs(FAR-FRR) == EER))));
%%%sprintf('L''Equal Error Rate (EER) è: %f con soglia: %f\n', EER, xEER)


ZeroFAR = max(find(FAR==0));
GARZeroFAR = 1-FRR(ZeroFAR);
% creiamo il grafico 1) FAR/FRR e EER
h=figure;
title('Performances');
subplot(2,2,1);
hold on;
grid on
box on

plot(x, FAR, 'r-');
plot(x, FRR, 'b-');
plot(xEER, EER, 'ko');

title('FAR/FRR for the System');
xlabel('FAR');
ylabel('FRR');
legend('FAR', 'FRR', 'EER');


% creiamo il grafico 2) disegnamo la curva ROC
subplot(2,2,2);
hold on;
grid on
box on
plot(FAR, 1-FRR, 'k-');

title('ROC Curve for the System');
xlabel('FAR');
ylabel('GAR');
legend('ROC Curve');

AUC = trapz(FAR, 1-FRR);


% Calcoliamo il CMS     

dist = zeros(1, gallery_dim); % vettore contenente le distanze fra un probe e tutte le gallery
CMS = zeros(1, gallery_dim);

Genuins = 0;
Impostors = 0;
gen_cnt = 1;
imp_cnt = 1;

AFF = 0; % numero di probe effettivi; quelli con DM(i,1)!=-1.
Identified = 0;
CMS_denom = 0;
for i=1:probe_dim
    
    PIsInG = find(Probe(i) == Gallery);  
    for h=1:length(PIsInG)
        Genuins(gen_cnt) = DM(i, PIsInG(h));
        gen_cnt = gen_cnt + 1;
    end
    
    PIsntInG = find(Probe(i) ~= Gallery);
    for h=1:length(PIsntInG)
        Impostors(imp_cnt) = DM(i, PIsntInG(h));
        imp_cnt = imp_cnt + 1;
    end
    
    if(DM(i,1) ~= -1)
        AFF = AFF + 1;
    end
    
    if(DM(i,1) ~= -1 && isempty(PIsInG)==0)
        
        CMS_denom = CMS_denom + 1;
        
        dist = DM(i,:);
    
        [Dist, I] = sort(dist);
        
        strt = 1;
        while (Dist(strt) == 0 && strt<gallery_dim)
            strt = strt + 1;
        end
        
        if(Probe(i)==Gallery(I(strt)))
            Identified(i) = 1;
        end
                
        for j=strt:gallery_dim    
              
           if (Probe(i)==Gallery(I(j))) %&& Dist(j)>0
               
               for k=j-strt+1:gallery_dim
                   CMS(k) = CMS(k) + 1;
               end

               break
           
           end
        
        end
    
    end
   
end


if AFF > 0
    CMS = CMS/CMS_denom;
else
    CMS = zeros(1, probe_dim);
end

% if(tipo==1)
%     inc = 1 - CMS; 
%     p = find(CMS<max(CMS));
%     inc = inc/length(p);
%     CMS = CMS + 2*inc;
%     p = find(CMS>1);
%     CMS(p) = 1.0;
% end

% creiamo il grafico 3) disegnamo la curva CMS
subplot(2,2,3);
hold on;
grid on
box on
plot(1:gallery_dim, CMS, 'k.-');

title('CMS Curve for the System');
xlabel('rank');
ylabel('CMS');
legend('CMS Curve');


RR = CMS(1);

% % calcoliamo la distribuzione degli scores per i genuini e gli impostori
PRECISION = 100;
scores = linspace(0, 1, PRECISION);
[Gen, Gcent] = hist(Genuins, PRECISION);
[Imp, Icent] = hist(Impostors, PRECISION);
Gen = Gen/max(Gen);
Imp = Imp/max(Imp);

% creiamo il grafico 4) disegnamo le curve con le distribuzioni degli score 
% per genuini ed impostori
subplot(2,2,4);
hold on;
grid on
box on
plot(Gcent, Gen, 'g');
plot(Icent, Imp, 'r');

areaGI = 0;
for i=1:PRECISION
    if ( Gen(i)>0 && Imp(i)>0)
        h = min([Gen(i) Imp(i)]);
        stem(scores(i), h, 'b');
        areaGI = areaGI + h;
    end
end

areaGI = double(areaGI / sum((Gen>0).*(Imp>0)));
    
title('Score distribution Genuins/Imposotrs');
xlabel('scores');
ylabel('probability density');
legend('Genuins', 'Impostors');

saveas(gcf, titolo, 'fig');

%%%sprintf('                               _\nArea di intersezione (Genuins | | Impostors): %f\n', areaGI)


                




