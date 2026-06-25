import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import pdb
torch.manual_seed(123)
class USCNN(nn.Module):
        def __init__(self,h,nx,ny,nVarIn=2,nVarOut=3,initWay=None,k=5,s=1,p=2):
                super(USCNN, self).__init__()
                """
                Extract basic information
                """
                self.initWay=initWay
                self.nVarIn=nVarIn
                self.nVarOut=nVarOut
                self.k=k
                self.s=1
                self.p=2
                self.deltaX=h
                self.nx=nx
                self.ny=ny


                """
                Define net
                """
                self.relu=nn.ReLU()
                self.US=nn.Upsample(size=[self.ny-2,self.nx-2],mode='bicubic')
                self.conv1=nn.Conv2d(self.nVarIn,16,kernel_size=k, stride=s, padding=p)
                self.conv2=nn.Conv2d(16,32,kernel_size=k, stride=s, padding=p)
                self.conv3=nn.Conv2d(32,16,kernel_size=k, stride=s, padding=p)
                self.conv4=nn.Conv2d(16,self.nVarOut,kernel_size=k, stride=s, padding=p)
                self.pixel_shuffle = nn.PixelShuffle(1)
                if self.initWay is not None:
                        self._initialize_weights()

                dxFilter=torch.Tensor([[[[0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.],
                       [1., -8.,  0.,  8.,  -1.],
                       [0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.]]]]).to("cuda")/12./self.deltaX
                self.convdx=nn.Conv2d(1, 1, (5,5),stride=1, padding=0, bias=None)
                self.convdx.weight=nn.Parameter(dxFilter, requires_grad=False)

                dyFilter=torch.Tensor([[[[0.,  0.,  1., 0.,  0.],
                       [0.,  0.,  -8.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  8., 0.,  0.],
                       [0.,  0.,  -1.,  0.,  0.]]]]).to("cuda")/12./self.deltaX
                self.convdy=nn.Conv2d(1,1,(5,5),stride=1,padding=0,bias=None)
                self.convdy.weight=nn.Parameter(dyFilter,requires_grad=False)

                lapFilter=torch.Tensor([[[[0.,  0.,  -1.,  0.,   0.],
                        [0.,  0.,  16.,  0.,   0.],
                        [-1., 16., -60., 16., -1.],
                        [0.,  0.,  16.,  0.,   0.],
                        [0.,  0.,  -1.,  0.,   0.]]]]).to("cuda")/12./self.deltaX/self.deltaX
                self.convlap = nn.Conv2d(1, 1, (5,5),stride=1, padding=0, bias=None)
                self.convlap.weight=nn.Parameter(lapFilter, requires_grad=False)

        def forward(self, x):
                x=self.US(x)
                x=self.relu(self.conv1(x))
                x=self.relu(self.conv2(x))
                x=self.relu(self.conv3(x))
                x=self.pixel_shuffle(self.conv4(x))

                return x

        def _initialize_weights(self):
                if self.initWay=='kaiming':
                        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv4.weight)
                elif self.initWay=='ortho':
                        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv4.weight)
                else:
                        print('Only Kaiming or Orthogonal initializer can be used!')
                        exit()


class VoronoiEnhancedUSCNNT(nn.Module):
        def __init__(self, h, nx, ny, nVarIn=2, nVarOut=3, initWay=None, k=5, s=1, p=2):
                super(VoronoiEnhancedUSCNNT, self).__init__()

                self.initWay = initWay
                self.nVarIn = nVarIn
                self.nVarOut = nVarOut
                self.k = k
                self.s = 1
                self.p = 2
                self.deltaX = h
                self.nx = nx
                self.ny = ny


                self.voronoi_branch = nn.Sequential(
                 nn.Conv2d(2, 32, kernel_size=k, stride=s, padding=p),
                 nn.ReLU(),
                 nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
                 nn.ReLU()
                )


                self.coord_branch = nn.Sequential(
                 nn.Conv2d(nVarIn, 32, kernel_size=k, stride=s, padding=p),
                 nn.ReLU(),
                 nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
                 nn.ReLU()
                )


                self.fusion_branch = nn.Sequential(
                 nn.Conv2d(64, 64, kernel_size=k, stride=s, padding=p),
                 nn.ReLU(),
                 nn.Conv2d(64, 32, kernel_size=k, stride=s, padding=p),
                 nn.ReLU(),
                 nn.Conv2d(32, nVarOut, kernel_size=k, stride=s, padding=p)
                )


                self.US = nn.Upsample(size=[self.ny - 2, self.nx - 2], mode='bicubic')
                self.pixel_shuffle = nn.PixelShuffle(1)


                dxFilter = torch.Tensor([[[[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [1., -8., 0., 8., -1.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]]]]).to("cuda") / 12. / self.deltaX
                self.convdx = nn.Conv2d(1, 1, (5, 5), stride=1, padding=0, bias=None)
                self.convdx.weight = nn.Parameter(dxFilter, requires_grad=False)

                dyFilter = torch.Tensor([[[[0., 0., 1., 0., 0.],
                         [0., 0., -8., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 8., 0., 0.],
                         [0., 0., -1., 0., 0.]]]]).to("cuda") / 12. / self.deltaX
                self.convdy = nn.Conv2d(1, 1, (5, 5), stride=1, padding=0, bias=None)
                self.convdy.weight = nn.Parameter(dyFilter, requires_grad=False)

                lapFilter = torch.Tensor([[[[0., 0., -1., 0., 0.],
                       [0., 0., 16., 0., 0.],
                       [-1., 16., -60., 16., -1.],
                       [0., 0., 16., 0., 0.],
                       [0., 0., -1., 0., 0.]]]]).to("cuda") / 12. / self.deltaX / self.deltaX
                self.convlap = nn.Conv2d(1, 1, (5, 5), stride=1, padding=0, bias=None)
                self.convlap.weight = nn.Parameter(lapFilter, requires_grad=False)


        def forward(self, coord, voronoi_input):


                voronoi_feat = self.voronoi_branch(voronoi_input)


                coord_feat = self.coord_branch(coord)


                fused = torch.cat([voronoi_feat, coord_feat], dim=1)
                output = self.fusion_branch(fused)


                output = self.US(output)
                output = self.pixel_shuffle(output)

                return output

class USCNNSep(nn.Module):
        def __init__(self,h,nx,ny,nVarIn=1,nVarOut=1,initWay=None,k=5,s=1,p=2):
                super(USCNNSep, self).__init__()
                """
                Extract basic information
                """
                self.initWay=initWay
                self.nVarIn=nVarIn
                self.nVarOut=nVarOut
                self.k=k
                self.s=1
                self.p=2
                self.deltaX=h
                self.nx=nx
                self.ny=ny
                """
                Define net
                """
                W1=16
                W2=32
                self.relu=nn.ReLU()
                self.US=nn.Upsample(size=[self.ny-2,self.nx-2],mode='bicubic')
                self.conv1=nn.Conv2d(self.nVarIn,W1,kernel_size=k, stride=s, padding=p)
                self.conv2=nn.Conv2d(W1,W2,kernel_size=k, stride=s, padding=p)
                self.conv3=nn.Conv2d(W2,W1,kernel_size=k, stride=s, padding=p)
                self.conv4=nn.Conv2d(W1,self.nVarOut,kernel_size=k, stride=s, padding=p)
                self.pixel_shuffle1 = nn.PixelShuffle(1)
                self.conv11=nn.Conv2d(self.nVarIn,W1,kernel_size=k, stride=s, padding=p)
                self.conv22=nn.Conv2d(W1,W2,kernel_size=k, stride=s, padding=p)
                self.conv33=nn.Conv2d(W2,W1,kernel_size=k, stride=s, padding=p)
                self.conv44=nn.Conv2d(W1,self.nVarOut,kernel_size=k, stride=s, padding=p)
                self.pixel_shuffle11 = nn.PixelShuffle(1)
                self.conv111=nn.Conv2d(self.nVarIn,W1,kernel_size=k, stride=s, padding=p)
                self.conv222=nn.Conv2d(W1,W2,kernel_size=k, stride=s, padding=p)
                self.conv333=nn.Conv2d(W2,W1,kernel_size=k, stride=s, padding=p)
                self.conv444=nn.Conv2d(W1,self.nVarOut,kernel_size=k, stride=s, padding=p)
                self.pixel_shuffle111 = nn.PixelShuffle(1)
                if self.initWay is not None:
                        self._initialize_weights()

                dxiFilter=torch.Tensor([[[[0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.],
                       [1., -8.,  0.,  8.,  -1.],
                       [0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.]]]]).to("cuda")/12./self.deltaX
                self.convdxi=nn.Conv2d(1, 1, (5,5),stride=1, padding=0, bias=None)
                self.convdxi.weight=nn.Parameter(dxiFilter, requires_grad=False)

                detaFilter=torch.Tensor([[[[0.,  0.,  1., 0.,  0.],
                       [0.,  0.,  -8.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  8., 0.,  0.],
                       [0.,  0.,  -1.,  0.,  0.]]]]).to("cuda")/12./self.deltaX
                self.convdeta=nn.Conv2d(1,1,(5,5),stride=1,padding=0,bias=None)
                self.convdeta.weight=nn.Parameter(detaFilter,requires_grad=False)

                lapFilter=torch.Tensor([[[[0.,  0.,  -1.,  0.,   0.],
                        [0.,  0.,  16.,  0.,   0.],
                        [-1., 16., -60., 16., -1.],
                        [0.,  0.,  16.,  0.,   0.],
                        [0.,  0.,  -1.,  0.,   0.]]]]).to("cuda")/12./self.deltaX/self.deltaX
                self.convlap = nn.Conv2d(1, 1, (5,5),stride=1, padding=0, bias=None)
                self.convlap.weight=nn.Parameter(lapFilter, requires_grad=False)

        def forward(self, x):
                x=self.US(x)
                x1=self.relu(self.conv1(x))
                x1=self.relu(self.conv2(x1))
                x1=self.relu(self.conv3(x1))
                x1=self.pixel_shuffle1(self.conv4(x1))

                x2=self.relu(self.conv11(x))
                x2=self.relu(self.conv22(x2))
                x2=self.relu(self.conv33(x2))
                x2=self.pixel_shuffle11(self.conv44(x2))

                x3=self.relu(self.conv111(x))
                x3=self.relu(self.conv222(x3))
                x3=self.relu(self.conv333(x3))
                x3=self.pixel_shuffle111(self.conv444(x3))
                return  torch.cat([x1,x2,x3],axis=1)


        def _initialize_weights(self):
                if self.initWay=='kaiming':
                        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv4.weight)
                        init.kaiming_normal_(self.conv11.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv22.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv33.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv44.weight)
                        init.kaiming_normal_(self.conv111.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv222.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv333.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv444.weight)
                elif self.initWay=='ortho':
                        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv4.weight)
                        init.orthogonal_(self.conv11.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv22.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv33.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv44.weight)
                        init.orthogonal_(self.conv111.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv222.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv333.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv444.weight)
                else:
                        print('Only Kaiming or Orthogonal initializer can be used!')
                        exit()

class VoronoiEnhancedUSCNNSep(nn.Module):
        def __init__(self,h,nx,ny,nVorCh=4,nVarIn=1,nVarOut=1,initWay=None,k=5,s=1,p=2):
                super(VoronoiEnhancedUSCNNSep, self).__init__()
                """
                Extract basic information
                """
                self.initWay=initWay
                self.nVarIn=nVarIn
                self.nVarOut=nVarOut
                self.k=k
                self.s=1
                self.p=2
                self.deltaX=h
                self.nx=nx
                self.ny=ny
                """
                Define net
                """
                Wc1, Wc2 = 32, 32
                Wv1, Wv2 = 32, 32
                Wf1, Wf2 = 64, 32

                W1=16
                W2=32
                self.relu=nn.ReLU()
                self.US=nn.Upsample(size=[self.ny-2,self.nx-2],mode='bicubic')


                self.voronoi_branch = nn.Sequential(
                 nn.Conv2d(nVorCh, Wv1, kernel_size=k, stride=s, padding=p),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(Wv1, Wv2, kernel_size=k, stride=s, padding=p),
                 nn.ReLU(inplace=True),
                )


                self.coord_branch = nn.Sequential(
                 nn.Conv2d(2, Wc1, kernel_size=k, stride=s, padding=p),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(Wc1, Wc2, kernel_size=k, stride=s, padding=p),
                 nn.ReLU(inplace=True),
                )


                self.fusion_branch = nn.Sequential(
                 nn.Conv2d(Wv2 + Wc2, 64, kernel_size=k, stride=s, padding=p),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(64, 32, kernel_size=k, stride=s, padding=p),
                 nn.ReLU(inplace=True),
                )


                self.convU1 = nn.Conv2d(32, 16, kernel_size=k, stride=s, padding=p)
                self.convU2 = nn.Conv2d(16, 32, kernel_size=k, stride=s, padding=p)
                self.convU3 = nn.Conv2d(32, 16, kernel_size=k, stride=s, padding=p)
                self.convU4 = nn.Conv2d(16, 1, kernel_size=k, stride=s, padding=p)
                self.psU = nn.PixelShuffle(1)


                self.convV1 = nn.Conv2d(32, 16, kernel_size=k, stride=s, padding=p)
                self.convV2 = nn.Conv2d(16, 32, kernel_size=k, stride=s, padding=p)
                self.convV3 = nn.Conv2d(32, 16, kernel_size=k, stride=s, padding=p)
                self.convV4 = nn.Conv2d(16, 1, kernel_size=k, stride=s, padding=p)
                self.psV = nn.PixelShuffle(1)


                self.convP1 = nn.Conv2d(32, 16, kernel_size=k, stride=s, padding=p)
                self.convP2 = nn.Conv2d(16, 32, kernel_size=k, stride=s, padding=p)
                self.convP3 = nn.Conv2d(32, 16, kernel_size=k, stride=s, padding=p)
                self.convP4 = nn.Conv2d(16, 1, kernel_size=k, stride=s, padding=p)
                self.psP = nn.PixelShuffle(1)


                dxiFilter=torch.Tensor([[[[0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.],
                       [1., -8.,  0.,  8.,  -1.],
                       [0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.]]]]).to("cuda")/12./self.deltaX
                self.convdxi=nn.Conv2d(1, 1, (5,5),stride=1, padding=0, bias=None)
                self.convdxi.weight=nn.Parameter(dxiFilter, requires_grad=False)

                detaFilter=torch.Tensor([[[[0.,  0.,  1., 0.,  0.],
                       [0.,  0.,  -8.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  8., 0.,  0.],
                       [0.,  0.,  -1.,  0.,  0.]]]]).to("cuda")/12./self.deltaX
                self.convdeta=nn.Conv2d(1,1,(5,5),stride=1,padding=0,bias=None)
                self.convdeta.weight=nn.Parameter(detaFilter,requires_grad=False)

                lapFilter=torch.Tensor([[[[0.,  0.,  -1.,  0.,   0.],
                        [0.,  0.,  16.,  0.,   0.],
                        [-1., 16., -60., 16., -1.],
                        [0.,  0.,  16.,  0.,   0.],
                        [0.,  0.,  -1.,  0.,   0.]]]]).to("cuda")/12./self.deltaX/self.deltaX
                self.convlap = nn.Conv2d(1, 1, (5,5),stride=1, padding=0, bias=None)
                self.convlap.weight=nn.Parameter(lapFilter, requires_grad=False)

                if self.initWay is not None:
                        self._initialize_weights()

        def forward(self, coord, voronoi_input):


                coord_rs = self.US(coord)
                voronoi_rs = self.US(voronoi_input)


                coord_feat = self.coord_branch(coord_rs)
                voronoi_feat = self.voronoi_branch(voronoi_rs)


                fused = torch.cat([coord_feat, voronoi_feat], dim=1)
                shared = self.fusion_branch(fused)


                xu = F.relu(self.convU1(shared))
                xu = F.relu(self.convU2(xu))
                xu = F.relu(self.convU3(xu))
                xu = self.psU(self.convU4(xu))


                xv = F.relu(self.convV1(shared))
                xv = F.relu(self.convV2(xv))
                xv = F.relu(self.convV3(xv))
                xv = self.psV(self.convV4(xv))


                xp = F.relu(self.convP1(shared))
                xp = F.relu(self.convP2(xp))
                xp = F.relu(self.convP3(xp))
                xp = self.psP(self.convP4(xp))

                out = torch.cat([xu, xv, xp], axis=1)
                return out


        def _initialize_weights(self):

                def kaiming(m):


                        if isinstance(m, nn.Conv2d):

                                if m in {self.convU4, self.convV4, self.convP4}:
                                        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                                else:
                                        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                                if m.bias is not None:
                                        nn.init.zeros_(m.bias)

                def ortho(m):
                        if isinstance(m, nn.Conv2d):
                                init.orthogonal_(m.weight, gain=init.calculate_gain('relu'))
                                if m.bias is not None: nn.init.zeros_(m.bias)

                if self.initWay == 'kaiming':
                        self.apply(kaiming)
                elif self.initWay == 'ortho':
                        self.apply(ortho)
                else:
                        pass


class USCNNSepPhi(nn.Module):
        def __init__(self,outputSize,nVarIn=1,nVarOut=1,initWay=None,k=5,s=1,p=2):
                super(USCNNSepPhi, self).__init__()
                """
                Extract basic information
                """
                self.initWay=initWay
                self.nVarIn=nVarIn
                self.nVarOut=nVarOut
                self.k=k
                self.s=1
                self.p=2
                self.deltaX=1/outputSize
                self.outputSize=outputSize

                """
                Define net
                """
                W1=16
                W2=32
                self.relu=nn.ReLU()
                self.US=nn.Upsample(size=[self.outputSize,self.outputSize],mode='bicubic')
                self.conv1=nn.Conv2d(self.nVarIn,W1,kernel_size=k, stride=s, padding=p)
                self.conv2=nn.Conv2d(W1,W2,kernel_size=k, stride=s, padding=p)
                self.conv3=nn.Conv2d(W2,W1,kernel_size=k, stride=s, padding=p)
                self.conv4=nn.Conv2d(W1,self.nVarOut,kernel_size=k, stride=s, padding=p)
                self.pixel_shuffle1 = nn.PixelShuffle(1)
                self.conv11=nn.Conv2d(self.nVarIn,W1,kernel_size=k, stride=s, padding=p)
                self.conv22=nn.Conv2d(W1,W2,kernel_size=k, stride=s, padding=p)
                self.conv33=nn.Conv2d(W2,W1,kernel_size=k, stride=s, padding=p)
                self.conv44=nn.Conv2d(W1,self.nVarOut,kernel_size=k, stride=s, padding=p)
                self.pixel_shuffle11 = nn.PixelShuffle(1)
                if self.initWay is not None:
                        self._initialize_weights()

                shrinkFilter=torch.Tensor([[[[0.,  0.,  0.,  0.,  0.],
                           [0.,  0.,  0.,  0.,  0.],
                         [0.,  0.,  1.,  0.,  0.],
                           [0.,  0.,  0.,  0.,  0.],
                           [0.,  0.,  0.,  0.,  0.]]]]).to("cuda")
                self.convShrink=nn.Conv2d(1, 1, (5,5),stride=1, padding=0, bias=None)
                self.convShrink.weight=nn.Parameter(shrinkFilter, requires_grad=False)


                dxFilter=torch.Tensor([[[[0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.],
                       [1., -8.,  0.,  8.,  -1.],
                       [0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.]]]]).to("cuda")/12./self.deltaX
                self.convdx=nn.Conv2d(1, 1, (5,5),stride=1, padding=0, bias=None)
                self.convdx.weight=nn.Parameter(dxFilter, requires_grad=False)

                dyFilter=torch.Tensor([[[[0.,  0.,  1., 0.,  0.],
                       [0.,  0.,  -8.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  8., 0.,  0.],
                       [0.,  0.,  -1.,  0.,  0.]]]]).to("cuda")/12./self.deltaX
                self.convdy=nn.Conv2d(1,1,(5,5),stride=1,padding=0,bias=None)
                self.convdy.weight=nn.Parameter(dyFilter,requires_grad=False)

                lapFilter=torch.Tensor([[[[0.,  0.,  -1.,  0.,   0.],
                        [0.,  0.,  16.,  0.,   0.],
                        [-1., 16., -60., 16., -1.],
                        [0.,  0.,  16.,  0.,   0.],
                        [0.,  0.,  -1.,  0.,   0.]]]]).to("cuda")/12./self.deltaX/self.deltaX
                self.convlap = nn.Conv2d(1, 1, (5,5),stride=1, padding=0, bias=None)
                self.convlap.weight=nn.Parameter(lapFilter, requires_grad=False)

        def forward(self, x):
                x=self.US(x)
                x1=self.relu(self.conv1(x))
                x1=self.relu(self.conv2(x1))
                x1=self.relu(self.conv3(x1))
                x1=self.pixel_shuffle1(self.conv4(x1))

                x2=self.relu(self.conv11(x))
                x2=self.relu(self.conv22(x2))
                x2=self.relu(self.conv33(x2))
                x2=self.pixel_shuffle11(self.conv44(x2))

                return  torch.cat([x1,x2],axis=1)

        def _initialize_weights(self):
                if self.initWay=='kaiming':
                        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv4.weight)
                        init.kaiming_normal_(self.conv11.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv22.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv33.weight, mode='fan_out', nonlinearity='relu')
                        init.kaiming_normal_(self.conv44.weight)
                elif self.initWay=='ortho':
                        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv4.weight)
                        init.orthogonal_(self.conv11.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv22.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv33.weight, init.calculate_gain('relu'))
                        init.orthogonal_(self.conv44.weight)
                else:
                        print('Only Kaiming or Orthogonal initializer can be used!')
                        exit()


def flatchannel(outputSize,nVarIn,nVarOut,W1,W2,k,s,p):
        return nn.Sequential(nn.Upsample(size=[int(outputSize/5),int(outputSize/5)],mode='bicubic'),
                nn.Conv2d(nVarIn,W1,kernel_size=k, stride=s, padding=p),
                nn.ReLU(),
                      nn.Conv2d(W1,W2,kernel_size=k, stride=s, padding=p),
                      nn.ReLU(),
                      nn.Conv2d(W2,W1,kernel_size=k, stride=s, padding=p),
                      nn.ReLU(),
                      nn.Conv2d(W1,nVarOut,kernel_size=k, stride=s, padding=p),
                      nn.PixelShuffle(1)).to('cuda')


class DDBasic(nn.Module):

        def __init__(self,outputSize,
                       nVarIn=1,
                       nVarOut=1,
                       initWay=None,
                       k=5,s=1,p=2):
                super(DDBasic,self).__init__()
                self.initWay=initWay
                self.nVarIn=nVarIn
                self.nVarOut=nVarOut
                self.k=k
                self.s=1
                self.p=2
                self.deltaX=1/outputSize
                self.outputSize=outputSize

                W1=16
                W2=32
                self.relu=nn.ReLU()
                self.US=nn.Upsample(size=[self.outputSize,self.outputSize],mode='bicubic')
                self.DS=nn.Upsample(size=[int(self.outputSize/2),int(self.outputSize/2)],mode='bicubic')


                for i in range(25):
                        exec("self.Phi_"+str(int(i))+"=flatchannel(self.outputSize,self.nVarIn,self.nVarOut,W1,W2,self.k,self.s,self.p)")
                        exec("self.P_"+str(int(i))+"=flatchannel(self.outputSize,self.nVarIn,self.nVarOut,W1,W2,self.k,self.s,self.p)")
                if self.initWay is not None:
                        self._initialize_weights()
                shrinkFilter=torch.Tensor([[[[0.,  0.,  0.,  0.,  0.],
                           [0.,  0.,  0.,  0.,  0.],
                         [0.,  0.,  1.,  0.,  0.],
                           [0.,  0.,  0.,  0.,  0.],
                           [0.,  0.,  0.,  0.,  0.]]]]).to("cuda")
                self.convShrink=nn.Conv2d(1, 1, (5,5),stride=1, padding=0, bias=None)
                self.convShrink.weight=nn.Parameter(shrinkFilter, requires_grad=False)


                dxFilter=torch.Tensor([[[[0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.],
                       [1., -8.,  0.,  8.,  -1.],
                       [0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.]]]]).to("cuda")/12./self.deltaX
                self.convdx=nn.Conv2d(1, 1, (5,5),stride=1, padding=0, bias=None)
                self.convdx.weight=nn.Parameter(dxFilter, requires_grad=False)

                dyFilter=torch.Tensor([[[[0.,  0.,  1., 0.,  0.],
                       [0.,  0.,  -8.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  8., 0.,  0.],
                       [0.,  0.,  -1.,  0.,  0.]]]]).to("cuda")/12./self.deltaX
                self.convdy=nn.Conv2d(1,1,(5,5),stride=1,padding=0,bias=None)
                self.convdy.weight=nn.Parameter(dyFilter,requires_grad=False)

                lapFilter=torch.Tensor([[[[0.,  0.,  -1.,  0.,   0.],
                        [0.,  0.,  16.,  0.,   0.],
                        [-1., 16., -60., 16., -1.],
                        [0.,  0.,  16.,  0.,   0.],
                        [0.,  0.,  -1.,  0.,   0.]]]]).to("cuda")/12./self.deltaX/self.deltaX
                self.convlap = nn.Conv2d(1, 1, (5,5),stride=1, padding=0, bias=None)
                self.convlap.weight=nn.Parameter(lapFilter, requires_grad=False)

        def forward(self, x):
                xup=self.US(x)
                x1=torch.zeros(xup[:,0:1,:,:].shape).to('cuda')
                x2=torch.zeros(xup[:,1:2,:,:].shape).to('cuda')
                l=10
                for i in range(5):
                        for j in range(5):

                                exec("x1[:,0:1,i*l:(i+1)*l,j*l:(j+1)*l]="+"self.Phi_"+str(int(i*5+j))+"(xup[:,:,i*l:(i+1)*l,j*l:(j+1)*l])")
                                exec("x2[:,0:1,i*l:(i+1)*l,j*l:(j+1)*l]="+"self.P_"+str(int(i*5+j))+"(xup[:,:,i*l:(i+1)*l,j*l:(j+1)*l])")


                '''
                x1[:,0,0:int(self.outputSize/2),0:int(self.outputSize/2)]=self.Phi_1(x[:,:,0:int(self.outputSize/2),0:int(self.outputSize/2)])
                x1[:,0,0:int(self.outputSize/2),int(self.outputSize/2):self.outputSize]=self.Phi_2(x[:,:,0:int(self.outputSize/2),int(self.outputSize/2):self.outputSize])
                x1[:,0,int(self.outputSize/2):self.outputSize,0:int(self.outputSize/2)]=self.Phi_3(x[:,:,int(self.outputSize/2):self.outputSize,0:int(self.outputSize/2)])
                x1[:,0,int(self.outputSize/2):self.outputSize,int(self.outputSize/2):self.outputSize]=self.Phi_4(x[:,:,int(self.outputSize/2):self.outputSize,int(self.outputSize/2):self.outputSize])

                x2[:,0,0:int(self.outputSize/2),0:int(self.outputSize/2)]=self.P_1(x[:,:,0:int(self.outputSize/2),0:int(self.outputSize/2)])
                x2[:,0,0:int(self.outputSize/2),int(self.outputSize/2):self.outputSize]=self.P_2(x[:,:,0:int(self.outputSize/2),int(self.outputSize/2):self.outputSize])
                x2[:,0,int(self.outputSize/2):self.outputSize,0:int(self.outputSize/2)]=self.P_3(x[:,:,int(self.outputSize/2):self.outputSize,0:int(self.outputSize/2)])
                x2[:,0,int(self.outputSize/2):self.outputSize,int(self.outputSize/2):self.outputSize]=self.P_4(x[:,:,int(self.outputSize/2):self.outputSize,int(self.outputSize/2):self.outputSize])
                '''
                return  torch.cat([x1,x2],axis=1)

        def _initialize_weights(self):
                if self.initWay=='kaiming':
                        if isinstance(m,nn.Conv2d):
                                init.kaiming_normal_(m.weight)
                elif self.initWay=='ortho':
                        if isinstance(m,nn.Conv2d):
                                init.kaiming_normal_(m.weight)
                else:
                        print('Only Kaiming or Orthogonal initializer can be used!')
                        exit()


class DDBasicSepNoPhi(nn.Module):

        def __init__(self,outputSize,
                       nVarIn=1,
                       nVarOut=1,
                       initWay=None,
                       k=5,s=1,p=2):
                super(DDBasicSepNoPhi,self).__init__()
                self.initWay=initWay
                self.nVarIn=nVarIn
                self.nVarOut=nVarOut
                self.k=k
                self.s=1
                self.p=2
                self.deltaX=1/outputSize
                self.outputSize=outputSize

                W1=16
                W2=32
                self.relu=nn.ReLU()
                self.US=nn.Upsample(size=[self.outputSize,self.outputSize],mode='bicubic')
                self.DS=nn.Upsample(size=[int(self.outputSize/2),int(self.outputSize/2)],mode='bicubic')


                for i in range(25):
                        exec("self.U_"+str(int(i))+"=flatchannel(self.outputSize,self.nVarIn,self.nVarOut,W1,W2,self.k,self.s,self.p)")
                        exec("self.V_"+str(int(i))+"=flatchannel(self.outputSize,self.nVarIn,self.nVarOut,W1,W2,self.k,self.s,self.p)")
                        exec("self.P_"+str(int(i))+"=flatchannel(self.outputSize,self.nVarIn,self.nVarOut,W1,W2,self.k,self.s,self.p)")
                if self.initWay is not None:
                        self._initialize_weights()
                shrinkFilter=torch.Tensor([[[[0.,  0.,  0.,  0.,  0.],
                           [0.,  0.,  0.,  0.,  0.],
                         [0.,  0.,  1.,  0.,  0.],
                           [0.,  0.,  0.,  0.,  0.],
                           [0.,  0.,  0.,  0.,  0.]]]]).to("cuda")
                self.convShrink=nn.Conv2d(1, 1, (5,5),stride=1, padding=0, bias=None)
                self.convShrink.weight=nn.Parameter(shrinkFilter, requires_grad=False)


                dxFilter=torch.Tensor([[[[0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.],
                       [1., -8.,  0.,  8.,  -1.],
                       [0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.]]]]).to("cuda")/12./self.deltaX
                self.convdx=nn.Conv2d(1, 1, (5,5),stride=1, padding=0, bias=None)
                self.convdx.weight=nn.Parameter(dxFilter, requires_grad=False)

                dyFilter=torch.Tensor([[[[0.,  0.,  1., 0.,  0.],
                       [0.,  0.,  -8.,  0.,  0.],
                       [0.,  0.,  0.,  0.,  0.],
                       [0.,  0.,  8., 0.,  0.],
                       [0.,  0.,  -1.,  0.,  0.]]]]).to("cuda")/12./self.deltaX
                self.convdy=nn.Conv2d(1,1,(5,5),stride=1,padding=0,bias=None)
                self.convdy.weight=nn.Parameter(dyFilter,requires_grad=False)

                lapFilter=torch.Tensor([[[[0.,  0.,  -1.,  0.,   0.],
                        [0.,  0.,  16.,  0.,   0.],
                        [-1., 16., -60., 16., -1.],
                        [0.,  0.,  16.,  0.,   0.],
                        [0.,  0.,  -1.,  0.,   0.]]]]).to("cuda")/12./self.deltaX/self.deltaX
                self.convlap = nn.Conv2d(1, 1, (5,5),stride=1, padding=0, bias=None)
                self.convlap.weight=nn.Parameter(lapFilter, requires_grad=False)

        def forward(self, x):
                xup=self.US(x)
                x1=torch.zeros(xup[:,0:1,:,:].shape).to('cuda')
                x2=torch.zeros(xup[:,0:1,:,:].shape).to('cuda')
                x3=torch.zeros(xup[:,0:1,:,:].shape).to('cuda')
                l=10
                for i in range(5):
                        for j in range(5):

                                exec("x1[:,0:1,i*l:(i+1)*l,j*l:(j+1)*l]="+"self.U_"+str(int(i*5+j))+"(xup[:,:,i*l:(i+1)*l,j*l:(j+1)*l])")
                                exec("x2[:,0:1,i*l:(i+1)*l,j*l:(j+1)*l]="+"self.V_"+str(int(i*5+j))+"(xup[:,:,i*l:(i+1)*l,j*l:(j+1)*l])")
                                exec("x3[:,0:1,i*l:(i+1)*l,j*l:(j+1)*l]="+"self.P_"+str(int(i*5+j))+"(xup[:,:,i*l:(i+1)*l,j*l:(j+1)*l])")


                '''
                x1[:,0,0:int(self.outputSize/2),0:int(self.outputSize/2)]=self.Phi_1(x[:,:,0:int(self.outputSize/2),0:int(self.outputSize/2)])
                x1[:,0,0:int(self.outputSize/2),int(self.outputSize/2):self.outputSize]=self.Phi_2(x[:,:,0:int(self.outputSize/2),int(self.outputSize/2):self.outputSize])
                x1[:,0,int(self.outputSize/2):self.outputSize,0:int(self.outputSize/2)]=self.Phi_3(x[:,:,int(self.outputSize/2):self.outputSize,0:int(self.outputSize/2)])
                x1[:,0,int(self.outputSize/2):self.outputSize,int(self.outputSize/2):self.outputSize]=self.Phi_4(x[:,:,int(self.outputSize/2):self.outputSize,int(self.outputSize/2):self.outputSize])

                x2[:,0,0:int(self.outputSize/2),0:int(self.outputSize/2)]=self.P_1(x[:,:,0:int(self.outputSize/2),0:int(self.outputSize/2)])
                x2[:,0,0:int(self.outputSize/2),int(self.outputSize/2):self.outputSize]=self.P_2(x[:,:,0:int(self.outputSize/2),int(self.outputSize/2):self.outputSize])
                x2[:,0,int(self.outputSize/2):self.outputSize,0:int(self.outputSize/2)]=self.P_3(x[:,:,int(self.outputSize/2):self.outputSize,0:int(self.outputSize/2)])
                x2[:,0,int(self.outputSize/2):self.outputSize,int(self.outputSize/2):self.outputSize]=self.P_4(x[:,:,int(self.outputSize/2):self.outputSize,int(self.outputSize/2):self.outputSize])
                '''
                return  torch.cat([x1,x2,x3],axis=1)

        def _initialize_weights(self):
                if self.initWay=='kaiming':
                        if isinstance(m,nn.Conv2d):
                                init.kaiming_normal_(m.weight)
                elif self.initWay=='ortho':
                        if isinstance(m,nn.Conv2d):
                                init.kaiming_normal_(m.weight)
                else:
                        print('Only Kaiming or Orthogonal initializer can be used!')
                        exit()
