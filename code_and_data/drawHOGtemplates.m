function im = drawHOGtemplates(winput, target_binsize)

flen = target_binsize(1) * target_binsize(2) * 32;

% wmaxx = max(0.0001, max(abs(winput(1:end-1))));
% wmaxx
% winput = winput ./ wmaxx;
w = winput(1:flen);
w = reshape(w, [target_binsize(2) target_binsize(1) 32]);

bs = 50;
% HOGpicture(w, bs)
% Make picture of positive HOG weights.

% construct a "glyph" for each orientaion
bim1 = zeros(bs, bs);
bim1(:,round(bs/2):round(bs/2)+1) = 1;
bim = zeros([size(bim1) 9]);
bim(:,:,1) = bim1;
for i = 2:9,
  bim(:,:,i) = imrotate(bim1, -(i-1)*20, 'crop');
end

% make pictures of positive weights bs adding up weighted glyphs
s = size(w);    
% w = abs(w);

wpos = w;
wpos(wpos < 0) = 0;    
im = zeros(bs*s(1), bs*s(2));
for i = 1:s(1),
  iis = (i-1)*bs+1:i*bs;
  for j = 1:s(2),
    jjs = (j-1)*bs+1:j*bs;          
    for k = 1:9,
      im(iis,jjs) = im(iis,jjs) + bim(:,:,k) * wpos(i,j,k);
    end
  end
end


end
