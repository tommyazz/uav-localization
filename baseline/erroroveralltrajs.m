clc
clear

cd ..
load('BScords.mat')
load('LOSindexmatrixforallBSalltrajs.mat')
cd 'Omni Spatial and Temporal Data'/
load("all_rxpower_tensor.mat")
load("all_toa_tensor.mat")
load("all_true_tensor.mat")
load("all_zenith_tensor.mat")
nsamps = 3e3;
nBS = 4;

est_xyz = zeros(50,3e3,3);
err_xyz = zeros(50,3e3);

warning on

 min_rx_cords = [373.5530  295.8490   68.1720];
    
    bs1cords = bs1cords - min_rx_cords ;
    bs2cords = bs2cords - min_rx_cords ;
    bs3cords = bs3cords - min_rx_cords ;
    bs4cords = bs4cords - min_rx_cords ;

for numtrj=1:50
    
    
    
    
   
    
    toa_matrix = reshape(toa_tensor(numtrj,:,:),[nsamps nBS]);
    
    rx_power_matrix = reshape(rx_power_tensor(numtrj,:,:),[nsamps nBS]);
    
    true_cord_matrix = reshape(true_cord_tensor(numtrj,:,:),[nsamps 3]);
    
    zenith_aoa_matrix = reshape(zenith_aoa_tensor(numtrj,:,:),[nsamps nBS]);
    
    
    for i = 1:3e3
        
        
        %toaarray = [ 8.6066e-07    7.3349e-07    3.4029e-07  5.0097e-07];
        
        %toaarray = [0.8178    0.1677    0.6273    0.8322]*1e-6;
        
        los_status = LOSidxmatrix(:,numtrj,i);
        
        toaarray = toa_matrix(i,:);
        rxpower = 10.^(0.1*rx_power_matrix(i,:));
        
        allrxpower = sum(rxpower);
        
        weightrxpower = rxpower/allrxpower;
        
        W = diag(weightrxpower);
        
        
%         if sum(los_status)>=3
%             w = zeros(1,4);
%             id_los = find(los_status);
%             w(id_los) = 1;
%          %   W = diag(w);
%             
%         end
        
        ri = toaarray*3e8 ;
        
        %Ri = [bs1cords(1)^2+bs1cords(2)^2 bs2cords(1)^2+bs2cords(2)^2 bs3cords(1)^2+bs3cords(2)^2 bs4cords(1)^2+bs4cords(2)^2];
        %
        %Ri = [norm(bs1cords)^2 norm(bs2cords)^2 norm(bs3cords)^2 norm(bs4cords)^2];
        
        Ri = [norm(bs1cords(1:2))^2 norm(bs2cords(1:2))^2 norm(bs3cords(1:2))^2 norm(bs4cords(1:2))^2];
        
        zenithangles =  zenith_aoa_matrix(i,:)*pi/180  ;
        
        h = ri.^2 - Ri;
        
        h=h';
        
        
        
        %G = [-2*bs1cords(1) -2*bs1cords(2) -2*bs1cords(3) 1;-2*bs2cords(1) -2*bs2cords(2) -2*bs2cords(3) 1;-2*bs3cords(1) -2*bs3cords(2) -2*bs3cords(3) 1;-2*bs4cords(1) -2*bs4cords(2) -2*bs4cords(3) 1];
        
        G = [-2*bs1cords(1) -2*bs1cords(2)  1;-2*bs2cords(1) -2*bs2cords(2)  1;-2*bs3cords(1) -2*bs3cords(2)  1;-2*bs4cords(1) -2*bs4cords(2)  1];
        
        
        %z = G\h;
        z = inv(G'*W*G)*G'*W*h;
        
        dist_2d = [(z(1) - bs1cords(1))^2 + (z(2) - bs1cords(2))^2;...
            (z(1) - bs2cords(1))^2 + (z(2) - bs2cords(2))^2;...
            (z(1) - bs3cords(1))^2 + (z(2) - bs3cords(2))^2;...
            (z(1) - bs4cords(1))^2 + (z(2) - bs4cords(2))^2];
        
        
        ht = sqrt(dist_2d').*(cos(zenithangles)./sin(zenithangles));
        
        height = -mean(ht - [bs1cords(3) bs2cords(3) bs3cords(3) bs4cords(3)]);
        
        z(3) = height;
        %error_xyz(i,:) = abs( (z(1:3))' -true_cord_matrix(i,1:3));
        
        est_xyz(numtrj,i,:) = z';
        
     %   err_xyz(numtrj,i) =  sqrt(sum(( (z(1:3))' -true_cord_matrix(i,1:3)).^2));
        err_xyz(numtrj,i) =  norm( (z(1:3))' -true_cord_matrix(i,1:3));
    end
    
    
end

ecdf(err_xyz(:))
axis([0 20 0 1])
grid on
