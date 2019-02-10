clc; clear; close all; currpath = cd; addpath(genpath(currpath));
%% --- loading ---
load 'data2.mat' %t=0ԭʼ����
[nAI,NUM_DMs]=size(data); %NOTE THAT:Decision Makers-DMs��numerical Assessment Information-nAI
% --- Para ---
n=9;   %��������
N=3;   %���Ը���
M=6;
m=30;
alpha=0.3;  %����г����ֵ-NEED alpha
Tmax=7;    %���ѭ������
%Q=0.28;      %Ⱥ��ͻ����ֵ -NEED theta
XI=[1/3,1/3,1/3]; % the weights of attributes
t=0;
E=1:m;  %��������߼���

u=0.4; % ������·���ʱ ԭd_k��������Ϣ����
r=1-u; % ������·���ʱ �� d_k�Ľ�ϵ�����ר�ҵ�������Ϣ����
% the parameters of solve the SR problem
lambda=1.e-2;
tolerance=1.e-3;

%% Spares representation
% --- Phase 1 : Conflict Relationship Investigation ---
D=data; % ��Ϊÿ�ε���һ��DD�е�ֵ һֱ�ڱ�� ������D����任�����е�
SR_coeffcients=zeros(NUM_DMs);
for m=1:NUM_DMs
    [y,data_]=AMR_Operator(data',m);
    [Coef, ~] = SolveHomotopy(data_',y','lambda',lambda,'tolerance',tolerance);
    SR_coeffcients(:,m)=Coef; %column
end
BETA=SR_coeffcients;
SR_coeffcients(SR_coeffcients>0)=0;
C=abs(SR_coeffcients);% Establish the conflict network




%% ---  Conflict relationship detection ---
% detect the opinion conflict or behavior conflict
% NOTE THAT OC means opinion conflict, BC means behavior conflict
OC=zeros(1,NUM_DMs);
BC=OC;
for iii=1:NUM_DMs
    BC(iii)=sum(C(:,iii));  %calculate \sum_j=1^I c(ij)
    OC(iii)=sum(C(iii,:));  %calculate \sum_j=1^I c(ji)
end


%% calculate AC and CD
AC=sum(OC)/NUM_DMs;
CD= 2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
Q=alpha*AC;  % set the conflict threshold

Operation=5;%Ϊ���ó������
while (Operation==5||Operation==6&&t<=Tmax&&~isempty(E))
    SR_coeffcients=zeros(NUM_DMs);
    D=DD;
    for i=1:NUM_DMs
        y=D(:,i);
        D_=D;
        D_(:,i)=0;                      %AMR Operator
        beta=SR(D_,y);
        SR_coeffcients(:,i)=beta;       %column
    end
    BETA=SR_coeffcients;
    SR_coeffcients(SR_coeffcients>0)=0;
    C=abs(SR_coeffcients);
    CD= 2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
    OC=zeros(1,NUM_DMs);
    BC=OC;
    for iii=1:NUM_DMs
        BC(iii)=sum(C(:,iii)); %calculate \sum_j=1^I c(ij)
        OC(iii)=sum(C(iii,:)); %calculate \sum_j=1^I c(ji)
    end
    %% �Ծ����ߵĳ�ͻ��Ϊ���з���
    Upsilon=[];Omega=[]; Psi=[];
    for j=1:NUM_DMs
        if OC(j)==0
            if BC(j)==0
                Upsilon=[Upsilon j];
            else
                Psi=[Psi j];
            end
        else
            if BC(j)==0
                Omega=[Omega j];
            elseif OC(j)>BC(j)
                Omega=[Omega j];
            elseif OC(j)<BC(j)
                Psi=[Psi j];
            elseif OC(j)==BC(j)  %%Ding add
                if OC(j)<=AC;
                    Omega=[Omega j];
                else
                    Psi=[Psi j];
                end
            end
        end
    end
    
    CA=zeros(nAI,1); %collective opinion
    omega=zeros(NUM_DMs,1); %omega is the weight vector of DMs
    %%��������ߵ�Ȩ��
    for iii=1:NUM_DMs
        omega(iii)=(1-OC(iii)/sum(OC)-BC(iii)/sum(BC))/(NUM_DMs-2); % compute every weight for each DM
    end
    %%����ȫ������Ϣ
    CA=D*omega;%��һ���ǻ��collective assessment CA
    
    
    CAM=zeros(M,N);  % restructured CA
    S=zeros(M,1); % the score vector
    disp(['CD=',num2str(CD)])
    disp(['Theta=',num2str(Q)])
    E=[Omega Psi]; %E��Omega��Psi�Ĳ���
    %���û�м��϶�Ӧ��end
    if CD<=Q %��Ҫ�ٷ�������ʱ���´洢p(t) D CA
        
        %disp(D)
        CAM=reshape(CA,M,N); %�ع�
        S=CAM*XI'; %��÷��������۵÷� ����������Section 5
        TTT=2*t;
        result(TT).Upsilon=Upsilon;
        result(TT).Omega=Omega;
        result(TT).Psi=Psi;
        result(TT).CA=CA;
        result(TT).S=S;
        result(TT).omega=omega;
        
        disp(['D=DD'])
        disp(['�����÷ֽ��Ϊ��S=', num2str(S')])
        break
    else %pt>Q û�дﵽ��ͻ��ֵʱ��������
        if t<=Tmax
            t=t+1;
            
            CAM=reshape(CA,M,N); %�ع�
            S=CAM*XI'; %��÷��������۵÷� ����������Section 5
            T=2*t-1;
            result(T).Upsilon=Upsilon;
            result(T).Omega=Omega;
            result(T).Psi=Psi;
            result(T).CA=CA;
            result(T).S=S;
            result(T).omega=omega;
            disp(['��',num2str(t),'��ѭ��'])
            
            [q, a]=size(Omega); %��Omega����DM�ĸ���a
            [w,b]=size(Psi);%��Psi����DM�ĸ���b
            [V, z]=size(Upsilon); %��Upsilon ����DM�ĸ���
            AAA=[];B=[];AA=[];BB=[];J=[];II=[];
            OCC=[];%��ӦOmega�����������BCֵ��DM�ĵ�����������
            BCC=[];%��ӦPsi�����������BCֵ��DM�ĵ�����������
            
            
            %�µĸ��·�����Psi��
            %[q, a]=size(Omega); %��Omega����DM�ĸ���a
            
            
            D=DD;%DD���¸�ֵ�ĳ�ʼ������Ϣ
            for j=1:b
                B=[B BC(Psi(j))]; %�洢Psi������DMs��BCֵ
                BB=[BB Psi(j)];  %�洢Psi�����е�DM�ı�ǩ
            end
            [~,E_q]=sort(B,'descend');  %��BC�����߰��ճ�ͻ�Ƚ��дӴ�С����
            d_k=BB(E_q(1)); %�ҳ�BC�г�ͻֵ���ľ�����
            disp(['Psi�г�ͻֵ����DM�ı�ǩ',num2str(d_k)]) %���d_k
            for j=1:b
                BCC=[BCC,r*(D(:,Psi(j)))+u*(D(:,d_k))]; %���������ڵ���ƽ��
            end
            BCC(:,E_q(1))=[]; %ȥ���������
            BCC=[BCC, r*(CA)+u*(D(:,d_k))];%��CA��ƽ��
            for zz=1:z
                BCC=[BCC,r*(D(:,Upsilon(z)))+u*(D(:,d_k))]; %��û�г�ͻ��DM��ƽ��
            end
            
            BCC=[BCC sum(BCC,2)/(b+z)]; %��õ�BCC����ƽ��
            BCC=roundn(BCC(:,:), -1); %�������С�����һλ
            EE=[];%���º��������Ϣ ��ԭ��DM��������Ϣ�Ĳ�ֵ
            bpt=[];%�洢���º��CDֵ
            
            for jj=1:b+z+1
                
                %C(jj)=zeros(NUM_DMs,NUM_DMs);
                EE=[EE BCC(:,jj)-D(:,d_k)];%�����ֵ
                D(:,d_k)=BCC(:,jj);%����D �����Ǿ�������滻D�е�d_k�е�������Ϣ���CDֵ
                for i=1:NUM_DMs
                    y=D(:,i);
                    D_=D;
                    D_(:,i)=0;                      %AMR Operator
                    beta=SR(D_,y);
                    SR_coeffcients(:,i)=beta;       %column
                end
                BETA=SR_coeffcients;
                SR_coeffcients(SR_coeffcients>0)=0;
                C=abs(SR_coeffcients); %���ڴ洢ÿ���滻����õ�C���� ������һ���ļ��� d_k ͳһ���滻C��C��rb��
                %CC=C(jj);
                bp= 2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                
                bpt=[bpt bp];
                
            end
            
            
            
            [~,rb]=sort(bpt);%�����µõ���CD��С��������
            disp(['Psi���ϵĳ���b=',num2str(b)])
            disp(['Upsilon���ϵĳ���z=',num2str(z)])
            disp(['������ר����Psi�����е�λ��',num2str(E_q(1))])
            disp(['���ŵ���������ӦBCC������',num2str(rb(1))])
            disp(['Dk_after=',num2str(BCC(:,rb(1))')])
            disp(['CD_after=',num2str(bpt(rb(1)))])
            A=input('������Ϊ��ͻ����������Ը(YES����3,NO����4):');
            
            
            % �µĸ��·�����OC��
            while (A==3|isempty(Psi)&&~isempty(Omega)) %��Ҫ��OC �����������һ��end Ŀǰ��������while
                %������3������Psi��������Ĳ���
                DD(:,d_k)=BCC(:,rb(1));  %����D_after����������Omega�����н��е���
                
                %���¼���C����
                D=DD;
                for i=1:NUM_DMs
                    y=D(:,i);
                    D_=D;
                    D_(:,i)=0;                      %AMR Operator
                    beta=SR(D_,y);
                    SR_coeffcients(:,i)=beta;       %column
                end
                BETA=SR_coeffcients;
                SR_coeffcients(SR_coeffcients>0)=0;
                C=abs(SR_coeffcients); %�����C���ڼ�����һ���е�CD
                
                %CD=bpt(rb(1));
                CD=2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                
                AC=sum(sum(C(:,:)))/NUM_DMs;
                V=30*AC/CD;
                
                T=2*t-1;
                %% ��ż�����
                result(T).D=D;  %Ⱥ���߾���
                result(T).C=C;  %��ͻ�������
                result(T).k=d_k; %�������ľ����ߵı�ǩ
                result(T).CD=CD;  %Ⱥ���ͻ��
                result(T).AC=AC;
                result(T).CDD=CD;
                result(T).V=V;
                %result(T).d_k=BCC(:,rb(1));%��¼���·�����Ϣ
                % �µĸ��·�����OC��
                
                [q, a]=size(Omega); %��Omega����DM�ĸ���a
                %[w,b]=size(Psi);%��Psi����DM�ĸ���b
                %[V, z]=size(Upsilon); %��Upsilon ����DM�ĸ���
                for ii=1:a
                    AAA=[AAA OC(Omega(ii))]; %Omega�е�DM��Ӧ��OCֵ
                    AA=[AA Omega(ii)];   %Omega��DM��Ӧ�ı�ǩ
                end
                
                [~,E_q]=sort(AAA,'descend');  %��Omega�еľ����߰��ճ�ͻ�Ƚ��дӴ�С����
                d_k=AA(E_q(1)); %�ҳ�Omega�г�ͻֵ���ľ�����
                disp(['Omega�г�ͻֵ����DM�ı�ǩ',num2str(d_k)]) %���d_k
                
                for ii=1:a
                    OCC=[OCC,r*(D(:,Omega(ii)))+u*(D(:,d_k))];%��Omega�����е�����DM��������Ϣ��Ȩƽ��
                end
                
                OCC(:,E_q(1))=[];%ȥ���������
                OCC=[OCC, r*(CA)+u*(D(:,d_k))];%��CA��ƽ��
                for zz=1:z
                    OCC=[OCC,r*(D(:,Upsilon(z)))+u*(D(:,d_k))]; %��û�г�ͻ��DM��ƽ��
                end
                
                OCC=[OCC sum(OCC,2)/(a+z)]; %OCC ����������ƽ��
                OCC=roundn(OCC(:,:), -1); %����һλС��
                OEE=[];%OCC�����е�ÿһ����ԭʼd_k�еĲ�ֵ
                opt=[]; %����D������d_k����OCC��ÿһ�к�õ����µ�CD
                
                for ii=1:a+z+1
                    
                    OEE=[OEE OCC(:,ii)-D(:,d_k)];%�����ֵ
                    D(:,d_k)=OCC(:,ii);%����D ��d_k��Ȼ�����������º��CD
                    for i=1:NUM_DMs
                        y=D(:,i);
                        D_=D;
                        D_(:,i)=0;                      %AMR Operator
                        beta=SR(D_,y);
                        SR_coeffcients(:,i)=beta;       %column
                    end
                    BETA=SR_coeffcients;
                    SR_coeffcients(SR_coeffcients>0)=0;
                    C=abs(SR_coeffcients);
                    op= 2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                    
                    
                    opt=[opt op];
                    
                end
                [~,rb]=sort(opt);%�����µõ���CD��С��������
                disp(['Omega���ϵĳ���a=',num2str(a)])
                disp(['Upsilon���ϵĳ���z=',num2str(z)])
                disp(['������ר����Omega�����е�λ��',num2str(E_q(1))])
                disp(['���ŵ���������ӦOCC������',num2str(rb(1))])
                disp(['Dk_after=',num2str(OCC(:,rb(1))')])
                disp(['���º��Ⱥ���ͻ��CD=',num2str(opt(rb(1)))])
                A=input('����۵��ͻ����������Ը(YES����1,NO����2):');
                
                while(A==2) %Omega��DM�ܾ����ܵ���
                    
                    Omega(find(Omega==d_k))=[];  %����������߼��ϣ���ת���14�� %��ôת û�����ף�
                    %���¼���C����
                    D=DD;
                    
                    % �µĸ��·�����OC��
                    [q, a]=size(Omega); %��Omega����DM�ĸ���a
                    %[w,b]=size(Psi);%��Psi����DM�ĸ���b
                    %[V, z]=size(Upsilon); %��Upsilon ����DM�ĸ���
                    AAA=[];
                    AA=[];
                    for ii=1:a
                        AAA=[AAA OC(Omega(ii))]; %Omega�е�DM��Ӧ��OCֵ
                        AA=[AA Omega(ii)];   %Omega��DM��Ӧ�ı�ǩ
                    end
                    
                    [~,E_q]=sort(AAA,'descend');  %��Omega�еľ����߰��ճ�ͻ�Ƚ��дӴ�С����
                    d_k=AA(E_q(1)); %�ҳ�Omega�г�ͻֵ���ľ�����
                    disp(['Omega�г�ͻֵ����DM�ı�ǩ',num2str(d_k)]) %���d_k
                    
                    for ii=1:a
                        OCC=[OCC,r*(D(:,Omega(ii)))+u*(D(:,d_k))];%��Omega�����е�����DM��������Ϣ��Ȩƽ��
                    end
                    
                    OCC(:,E_q(1))=[];%ȥ���������
                    OCC=[OCC, r*(CA)+u*(D(:,d_k))];%��CA��ƽ��
                    for zz=1:z
                        OCC=[OCC,r*(D(:,Upsilon(z)))+u*(D(:,d_k))]; %��û�г�ͻ��DM��ƽ��
                    end
                    
                    OCC=[OCC sum(OCC,2)/(a+z)]; %OCC ����������ƽ��
                    OCC=roundn(OCC(:,:), -1); %����һλС��
                    OEE=[];%OCC�����е�ÿһ����ԭʼd_k�еĲ�ֵ
                    opt=[]; %����D������d_k����OCC��ÿһ�к�õ����µ�CD
                    
                    for ii=1:a+z+1
                        
                        OEE=[OEE OCC(:,ii)-D(:,d_k)];%�����ֵ
                        D(:,d_k)=OCC(:,ii);%����D ��d_k��Ȼ�����������º��CD
                        for i=1:NUM_DMs
                            y=D(:,i);
                            D_=D;
                            D_(:,i)=0;                      %AMR Operator
                            beta=SR(D_,y);
                            SR_coeffcients(:,i)=beta;       %column
                        end
                        BETA=SR_coeffcients;
                        SR_coeffcients(SR_coeffcients>0)=0;
                        C=abs(SR_coeffcients);
                        op= 2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                        
                        
                        opt=[opt op];
                        
                    end
                    [~,rb]=sort(opt);%�����µõ���CD��С��������
                    disp(['Omega���ϵĳ���a=',num2str(a)])
                    disp(['Upsilon���ϵĳ���z=',num2str(z)])
                    disp(['������ר����Omega�����е�λ��',num2str(E_q(1))])
                    disp(['���ŵ���������ӦOCC������',num2str(rb(1))])
                    disp(['Dk_after=',num2str(OCC(:,rb(1))')])
                    disp(['���º��Ⱥ���ͻ��CD=',num2str(opt(rb(1)))])
                    A=input('����۵��ͻ����������Ը(YES����1,NO����2):');
                    
                    
                end
                
                while(A==1)
                    DD(:,d_k)=OCC(:,rb(1));  %����D_after����ת������5
                    E=[Omega Psi];
                    %���¼���C����
                    D=DD;
                    for i=1:NUM_DMs
                        y=D(:,i);
                        D_=D;
                        D_(:,i)=0;                      %AMR Operator
                        beta=SR(D_,y);
                        SR_coeffcients(:,i)=beta;       %column
                    end
                    BETA=SR_coeffcients;
                    SR_coeffcients(SR_coeffcients>0)=0;
                    C=abs(SR_coeffcients); %�����C���ڼ�����һ���е�CD
                    CD=2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                    %CD=opt(rb(1));
                    AC=sum(sum(C(:,:)))/NUM_DMs;
                    V=30*AC/CD;
                    
                    %% ��ż�����
                    TT=2*t;
                    result(TT).D=D;  %Ⱥ���߾���
                    result(TT).C=C;  %��ͻ�������
                    result(TT).k=d_k; %�������ľ����ߵı�ǩ
                    result(TT).CD=CD;  %Ⱥ���ͻ��
                    %result(TT).d_k=OCC(:,rb(1));%��¼���µ�������Ϣ
                    result(TT).AC=AC;
                    result(TT).CDD=CD;
                    result(TT).V=V;
                    
                    %D=DD;%���·���
                    break
                    %Operation=input('����ѭ�������Ƿ����(YES����5,NO����6):');
                end %while=1 end
                
            end%��ӦOmega������while A3
            
            %while =4
            while (A==4) %Psi��d_k�ܾ���������
                
                Psi(find(Psi==d_k))=[];  %������Psi�����߼��ϣ���ת���14�� %��ôת --ͨ��whileѭ�����ת
                [q, a]=size(Omega); %��Omega����DM�ĸ���a
                [w,b]=size(Psi);%��Psi����DM�ĸ���b
                [V, z]=size(Upsilon); %��Upsilon ����DM�ĸ���
                B=[];
                BB=[];
                D=DD;%DD���¸�ֵ�ĳ�ʼ������Ϣ
                for j=1:b
                    B=[B BC(Psi(j))]; %�洢Psi������DMs��BCֵ
                    BB=[BB Psi(j)];  %�洢Psi�����е�DM�ı�ǩ
                end
                [~,E_q]=sort(B,'descend');  %��BC�����߰��ճ�ͻ�Ƚ��дӴ�С����
                d_k=BB(E_q(1)); %�ҳ�BC�г�ͻֵ���ľ�����
                disp(['Psi�г�ͻֵ����DM�ı�ǩ',num2str(d_k)]) %���d_k
                
                for j=1:b
                    BCC=[BCC,r*(D(:,Psi(j)))+u*(D(:,d_k))]; %���������ڵ���ƽ��
                end
                BCC(:,E_q(1))=[]; %ȥ��������� %weile shiyan
                BCC=[BCC, r*(CA)+u*(D(:,d_k))];%��CA��ƽ��
                for zz=1:z
                    BCC=[BCC,r*(D(:,Upsilon(z)))+u*(D(:,d_k))]; %��û�г�ͻ��DM��ƽ��
                end
                
                BCC=[BCC sum(BCC,2)/(b+z)]; %��õ�BCC����ƽ��
                BCC=roundn(BCC(:,:), -1); %�������С�����һλ
                EE=[];%���º��������Ϣ ��ԭ��DM��������Ϣ�Ĳ�ֵ
                bpt=[];%�洢���º��CDֵ
                
                for jj=1:b+z+1
                    
                    %C(jj)=zeros(NUM_DMs,NUM_DMs);
                    EE=[EE BCC(:,jj)-D(:,d_k)];%�����ֵ
                    D(:,d_k)=BCC(:,jj);%����D �����Ǿ�������滻D�е�d_k�е�������Ϣ���CDֵ
                    for i=1:NUM_DMs
                        y=D(:,i);
                        D_=D;
                        D_(:,i)=0;                      %AMR Operator
                        beta=SR(D_,y);
                        SR_coeffcients(:,i)=beta;       %column
                    end
                    BETA=SR_coeffcients;
                    SR_coeffcients(SR_coeffcients>0)=0;
                    C=abs(SR_coeffcients); %���ڴ洢ÿ���滻����õ�C���� ������һ���ļ��� d_k ͳһ���滻C��C��rb��
                    %CC=C(jj);
                    bp= 2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                    
                    bpt=[bpt bp];
                    
                end
                [~,rb]=sort(bpt);%�����µõ���CD��С��������
                disp(['Psi���ϵĳ���b=',num2str(b)])
                disp(['Upsilon���ϵĳ���z=',num2str(z)])
                disp(['������ר����Psi�����е�λ��',num2str(E_q(1))])
                disp(['���ŵ���������ӦBCC������',num2str(rb(1))])
                disp(['Dk_after=',num2str(BCC(:,rb(1))')])
                disp(['CD_after=',num2str(bpt(rb(1)))])
                A=input('������Ϊ��ͻ����������Ը(YES����3,NO����4):');
                
                
                while (A==3|isempty(Psi)&&~isempty(Omega)) %��Ҫ��OC �����������һ��end Ŀǰ��������while
                    %������3������Psi��������Ĳ���
                    DD(:,d_k)=BCC(:,rb(1));  %����D_after����������Omega�����н��е���
                    
                    %���¼���C����
                    D=DD;
                    for i=1:NUM_DMs
                        y=D(:,i);
                        D_=D;
                        D_(:,i)=0;                      %AMR Operator
                        beta=SR(D_,y);
                        SR_coeffcients(:,i)=beta;       %column
                    end
                    BETA=SR_coeffcients;
                    SR_coeffcients(SR_coeffcients>0)=0;
                    C=abs(SR_coeffcients); %�����C���ڼ�����һ���е�CD
                    
                    CD=2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                    %CD=bpt(rb(1));
                    AC=sum(sum(C(:,:)))/NUM_DMs;
                    V=30*AC/CD;
                    
                    %% ��ż�����
                    T=2*t-1;
                    result(T).D=D;  %Ⱥ���߾���
                    result(T).C=C;  %��ͻ�������
                    result(T).dk=d_k; %�������ľ����ߵı�ǩ
                    result(T).CD=CD;  %Ⱥ���ͻ��
                    result(T).AC=AC;
                    result(T).CDD=CD;
                    result(T).V=V;                    % �µĸ��·�����OC��
                    [q, a]=size(Omega); %��Omega����DM�ĸ���a
                    %[w,b]=size(Psi);%��Psi����DM�ĸ���b
                    %[V, z]=size(Upsilon); %��Upsilon ����DM�ĸ���
                    for ii=1:a
                        AAA=[AAA OC(Omega(ii))]; %Omega�е�DM��Ӧ��OCֵ
                        AA=[AA Omega(ii)];   %Omega��DM��Ӧ�ı�ǩ
                    end
                    
                    [~,E_q]=sort(AAA,'descend');  %��Omega�еľ����߰��ճ�ͻ�Ƚ��дӴ�С����
                    d_k=AA(E_q(1)); %�ҳ�Omega�г�ͻֵ���ľ�����
                    disp(['Omega�г�ͻֵ����DM�ı�ǩ',num2str(d_k)]) %���d_k
                    
                    for ii=1:a
                        OCC=[OCC,r*(D(:,Omega(ii)))+u*(D(:,d_k))];%��Omega�����е�����DM��������Ϣ��Ȩƽ��
                    end
                    
                    OCC(:,E_q(1))=[];%ȥ���������
                    OCC=[OCC, r*(CA)+u*(D(:,d_k))];%��CA��ƽ��
                    for zz=1:z
                        OCC=[OCC,r*(D(:,Upsilon(z)))+u*(D(:,d_k))]; %��û�г�ͻ��DM��ƽ��
                    end
                    
                    OCC=[OCC sum(OCC,2)/(a+z)]; %OCC ����������ƽ��
                    OCC=roundn(OCC(:,:), -1); %����һλС��
                    OEE=[];%OCC�����е�ÿһ����ԭʼd_k�еĲ�ֵ
                    opt=[]; %����D������d_k����OCC��ÿһ�к�õ����µ�CD
                    
                    for ii=1:a+z+1
                        
                        OEE=[OEE OCC(:,ii)-D(:,d_k)];%�����ֵ
                        D(:,d_k)=OCC(:,ii);%����D ��d_k��Ȼ�����������º��CD
                        for i=1:NUM_DMs
                            y=D(:,i);
                            D_=D;
                            D_(:,i)=0;                      %AMR Operator
                            beta=SR(D_,y);
                            SR_coeffcients(:,i)=beta;       %column
                        end
                        BETA=SR_coeffcients;
                        SR_coeffcients(SR_coeffcients>0)=0;
                        C=abs(SR_coeffcients);
                        op= 2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                        
                        
                        opt=[opt op];
                        
                    end
                    [~,rb]=sort(opt);%�����µõ���CD��С��������
                    disp(['Omega���ϵĳ���a=',num2str(a)])
                    disp(['Upsilon���ϵĳ���z=',num2str(z)])
                    disp(['������ר����Omega�����е�λ��',num2str(E_q(1))])
                    disp(['���ŵ���������ӦOCC������',num2str(rb(1))])
                    disp(['Dk_after=',num2str(OCC(:,rb(1))')])
                    disp(['���º��Ⱥ���ͻ��CD=',num2str(opt(rb(1)))])
                    A=input('����۵��ͻ����������Ը(YES����1,NO����2):');
                    
                    while(A==2) %Omega��DM�ܾ����ܵ���
                        Omega(find(Omega==d_k))=[];  %����������߼��ϣ���ת���14�� %��ôת û�����ף�
                        %���¼���C����
                        D=DD;
                        
                        % �µĸ��·�����OC��
                        [q, a]=size(Omega); %��Omega����DM�ĸ���a
                        %[w,b]=size(Psi);%��Psi����DM�ĸ���b
                        %[V, z]=size(Upsilon); %��Upsilon ����DM�ĸ���
                        AAA=[];
                        AA=[];
                        for ii=1:a
                            AAA=[AAA OC(Omega(ii))]; %Omega�е�DM��Ӧ��OCֵ
                            AA=[AA Omega(ii)];   %Omega��DM��Ӧ�ı�ǩ
                        end
                        
                        [~,E_q]=sort(AAA,'descend');  %��Omega�еľ����߰��ճ�ͻ�Ƚ��дӴ�С����
                        d_k=AA(E_q(1)); %�ҳ�Omega�г�ͻֵ���ľ�����
                        disp(['Omega�г�ͻֵ����DM�ı�ǩ',num2str(d_k)]) %���d_k
                        
                        for ii=1:a
                            OCC=[OCC,r*(D(:,Omega(ii)))+u*(D(:,d_k))];%��Omega�����е�����DM��������Ϣ��Ȩƽ��
                        end
                        
                        OCC(:,E_q(1))=[];%ȥ���������
                        OCC=[OCC, r*(CA)+u*(D(:,d_k))];%��CA��ƽ��
                        for zz=1:z
                            OCC=[OCC,r*(D(:,Upsilon(z)))+u*(D(:,d_k))]; %��û�г�ͻ��DM��ƽ��
                        end
                        
                        OCC=[OCC sum(OCC,2)/(a+z)]; %OCC ����������ƽ��
                        OCC=roundn(OCC(:,:), -1); %����һλС��
                        OEE=[];%OCC�����е�ÿһ����ԭʼd_k�еĲ�ֵ
                        opt=[]; %����D������d_k����OCC��ÿһ�к�õ����µ�CD
                        
                        for ii=1:a+z+1
                            
                            OEE=[OEE OCC(:,ii)-D(:,d_k)];%�����ֵ
                            D(:,d_k)=OCC(:,ii);%����D ��d_k��Ȼ�����������º��CD
                            for i=1:NUM_DMs
                                y=D(:,i);
                                D_=D;
                                D_(:,i)=0;                      %AMR Operator
                                beta=SR(D_,y);
                                SR_coeffcients(:,i)=beta;       %column
                            end
                            BETA=SR_coeffcients;
                            SR_coeffcients(SR_coeffcients>0)=0;
                            C=abs(SR_coeffcients);
                            op= 2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                            
                            
                            opt=[opt op];
                            
                        end
                        [~,rb]=sort(opt);%�����µõ���CD��С��������
                        disp(['Omega���ϵĳ���a=',num2str(a)])
                        disp(['Upsilon���ϵĳ���z=',num2str(z)])
                        disp(['������ר����Omega�����е�λ��',num2str(E_q(1))])
                        disp(['���ŵ���������ӦOCC������',num2str(rb(1))])
                        disp(['Dk_after=',num2str(OCC(:,rb(1))')])
                        disp(['���º��Ⱥ���ͻ��CD=',num2str(opt(rb(1)))])
                        A=input('����۵��ͻ����������Ը(YES����1,NO����2):');
                        
                        
                    end
                    
                    while(A==1)
                        DD(:,d_k)=OCC(:,rb(1));  %����D_after����ת������5
                        E=[Omega Psi];
                        %���¼���C����
                        D=DD;
                        for i=1:NUM_DMs
                            y=D(:,i);
                            D_=D;
                            D_(:,i)=0;                      %AMR Operator
                            beta=SR(D_,y);
                            SR_coeffcients(:,i)=beta;       %column
                        end
                        BETA=SR_coeffcients;
                        SR_coeffcients(SR_coeffcients>0)=0;
                        C=abs(SR_coeffcients); %�����C���ڼ�����һ���е�CD
                        
                        %CD=opt(rb(1));
                        CD=2*sum(sum(C(:,:)~=0))/(NUM_DMs*(NUM_DMs-1)); % group conflict degree CD
                        AC=sum(sum(C(:,:)))/NUM_DMs;
                        V=30*AC/CD;
                        
                        %% ��ż�����
                        TT=2*t;
                        result(TT).D=D;  %Ⱥ���߾���
                        result(TT).C=C;  %��ͻ�������
                        result(TT).dk=d_k; %�������ľ����ߵı�ǩ
                        result(TT).CD=CD;  %Ⱥ���ͻ��
                        result(TT).AC=AC;
                        result(TT).CDD=CD;
                        result(TT).V=V;                    %D=DD;%���·���
                        break
                        %Operation=input('����ѭ�������Ƿ����(YES����5,NO����6):');
                    end %while=1 end
                    
                end%��ӦOmega������while A3
                
            end%��ӦOmega������while A=4
            
            
        end%�����Ӧif t<=Tmax ����һ��CRP�ĵ���
        
    end
    % disp('FAILED')
    Operation=input('����ѭ�������Ƿ����(YES����5,NO����6):');
    
end
%Operation=input('����ѭ�������Ƿ����(YES����5,NO����6):');

























