function strimage(path, n)
  fidin = fopen(path); % ��train-01-images.svm�ļ�
  i = 1; % ��ʼ��������

  apres = []; % �������������ڴ洢����

  while ~feof(fidin) % ��δ�����ļ�ĩβʱѭ��
    tline = fgetl(fidin); % ���ļ��ж�ȡһ������
    apres{i} = tline; % �����ݴ�������
    i = i+1; % ����������
  end

  a = char(apres(n)); % ��ȡ�����е� n ������

  lena = size(a); % ��ȡ�ַ�������
  lena = lena(2); % ��ȡ�ַ����ĵڶ�ά�ȳ���

  xy = sscanf(a(4:lena), '%d:%d'); % ���ַ�������ȡ����

  lenxy = size(xy); % ��ȡ��ȡ�����ݳ���
  lenxy = lenxy(1); % ��ȡ���ݵĵ�һά�ȳ���

  grid = []; % �������������ڴ洢��������
  grid(784) = 0; % ��ʼ�������С

  for i=2:2:lenxy  % �ӵڶ�������ʼ��ÿ��һ����ȡֵ
      if(xy(i)<=0) % ���ֵС�ڵ���0������ѭ��
          break
      end
    grid(xy(i-1)) = xy(i) * 100/255; % ��������ֵ����������
  end

  grid1 = reshape(grid,28,28); % ��������Ϊ28x28�ľ���
  grid1 = fliplr(diag(ones(28,1)))*grid1; % �Ծ�����з�ת�ͶԽ��߲���
  grid1 = rot90(grid1,3); % ��ʱ����ת����

  image(grid1) % ��ʾͼ��
  hold on; % ����ͼ����ͬһ������
end
