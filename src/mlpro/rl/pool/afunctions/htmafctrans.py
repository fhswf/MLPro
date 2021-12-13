from statistics import mean
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import transformations

from collections import deque

from mlpro.rl.models import *


class transformationMatSingle(torch.nn.Module):
    def __init__(self, n_in, device):
        super(transformationMatSingle, self).__init__()
        self.n_in = n_in
        self.device = device
        self.secondNet = False
        self.added = 0

        self.register_parameter("aW", torch.nn.Parameter((torch.rand(self.n_in, 1, 1) - 0.5) * 1))
        self.register_parameter("uW", torch.nn.Parameter((torch.rand(self.n_in, 1, 3) - 0.5) * 1))
        self.register_parameter("tW", torch.nn.Parameter((torch.rand(self.n_in, 1, 3) - 0.5) * 1))

    def forward(self, I):
        BatchSize = I.shape[0]
        aV = self.aW.repeat(BatchSize, 1, 1).reshape(BatchSize, self.n_in, 1)
        uV = torch.sigmoid(self.uW.repeat(BatchSize, 1, 1)).reshape(BatchSize, self.n_in, 1, 3)
        tV = self.tW.repeat(BatchSize, 1, 1).reshape(BatchSize, self.n_in, 1, 3)
        saV = None
        suV = None
        stV = None
        maskingIndices = torch.tensor([[3, 2, 1], [2, 3, 0], [1, 0, 3]])
        maskingCross = torch.Tensor([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        fixVec = torch.Tensor([[0.0, 0.0, 0.0, 1.0]])
        out = torch.Tensor([])
        WM = None

        if self.secondNet:
            saV = self.saW.repeat(BatchSize, 1, 1).reshape(BatchSize, self.added, 1)
            suV = torch.sigmoid(self.suW.repeat(BatchSize, 1, 1)).reshape(BatchSize, self.added, 1, 3)
            stV = self.stW.repeat(BatchSize, 1, 1).reshape(BatchSize, self.added, 1, 3)

        for i in range(BatchSize):
            for j in range(self.n_in):
                U = uV[i][j]
                U = U / torch.norm(U)
                kunit = torch.cat([U, torch.Tensor([[0.0]])], dim=1)
                unit = kunit[0][maskingIndices] * maskingCross
                rotmat = (
                    unit * torch.sin(aV[i][j])
                    + (torch.eye(3) - torch.ger(U[0], U[0])) * torch.cos(aV[i][j])
                    + torch.ger(U[0], U[0])
                )
                stacked = torch.cat([rotmat, tV[i][j].T], dim=1)
                stacked = torch.cat([stacked, fixVec], dim=0)
                out = torch.cat([out, stacked])

            if self.secondNet:
                for j in range(self.added):
                    U = suV[i][j]
                    U = U / torch.norm(U)
                    kunit = torch.cat([U, torch.Tensor([[0.0]])], dim=1)
                    unit = kunit[0][maskingIndices] * maskingCross
                    rotmat = (
                        unit * torch.sin(saV[i][j])
                        + (torch.eye(3) - torch.ger(U[0], U[0])) * torch.cos(saV[i][j])
                        + torch.ger(U[0], U[0])
                    )
                    stacked = torch.cat([rotmat, stV[i][j].T], dim=1)
                    stacked = torch.cat([stacked, fixVec], dim=0)
                    out = torch.cat([out, stacked])

        if self.secondNet:
            WM = out.reshape(BatchSize, self.n_in + self.added, 4, 4)
        else:
            WM = out.reshape(BatchSize, self.n_in, 4, 4)

        O = torch.matmul(I, WM)

        return O

    def secondaryNetwork(self, jointadded):
        self.secondNet = True
        self.added = jointadded
        self.register_parameter("saW", torch.nn.Parameter((torch.rand(self.added, 1, 1) - 0.5) * 1))
        self.register_parameter("suW", torch.nn.Parameter((torch.rand(self.added, 1, 3) - 0.5) * 1))
        self.register_parameter("stW", torch.nn.Parameter((torch.rand(self.added, 1, 3) - 0.5) * 1))


class transformationMatMulti(torch.nn.Module):
    def __init__(self, n_in, n_out, device):
        super(transformationMatMulti, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.device = device

        self.register_parameter(
            "aW",
            torch.nn.Parameter((torch.rand(self.n_in * self.n_out, 1, 1) - 0.5) * 1),
        )
        self.register_parameter(
            "uW",
            torch.nn.Parameter((torch.rand(self.n_in * self.n_out, 1, 3) - 0.5) * 1),
        )
        self.register_parameter(
            "tW",
            torch.nn.Parameter((torch.rand(self.n_in * self.n_out, 1, 3) - 0.5) * 1),
        )

    def forward(self, I):
        BatchSize = I.shape[0]
        aV = self.aW.repeat(BatchSize, 1, 1).reshape(BatchSize, self.n_in * self.n_out, 1)
        uV = torch.sigmoid(self.uW.repeat(BatchSize, 1, 1)).reshape(BatchSize, self.n_in * self.n_out, 1, 3)
        tV = self.tW.repeat(BatchSize, 1, 1).reshape(BatchSize, self.n_in * self.n_out, 1, 3)
        maskingIndices = torch.tensor([[3, 2, 1], [2, 3, 0], [1, 0, 3]])
        maskingCross = torch.Tensor([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        fixVec = torch.Tensor([[0.0, 0.0, 0.0, 1.0]])
        out = torch.Tensor([])
        for i in range(BatchSize):
            for j in range(self.n_in * self.n_out):
                U = uV[i][j]
                U = U / torch.norm(U)
                kunit = torch.cat([U, torch.Tensor([[0.0]])], dim=1)
                unit = kunit[0][maskingIndices] * maskingCross
                rotmat = (
                    unit * torch.sin(aV[i][j])
                    + (torch.eye(3) - torch.ger(U[0], U[0])) * torch.cos(aV[i][j])
                    + torch.ger(U[0], U[0])
                )
                stacked = torch.cat([rotmat, tV[i][j].T], dim=1)
                stacked = torch.cat([stacked, fixVec], dim=0)
                out = torch.cat([out, stacked])

        WM = out.reshape(BatchSize, self.n_in, self.n_out, 4, 4)

        out = torch.Tensor([])
        for i in range(BatchSize):
            out1 = torch.eye(4).repeat(self.n_out, 1, 1)
            for j in range(self.n_in):
                out2 = torch.Tensor([])
                for k in range(self.n_out):
                    out2 = torch.cat([out2, torch.matmul(I[i][j], WM[i][j][k])])
                out1 = torch.matmul(out1, out2.reshape(self.n_out, 4, 4))
            out = torch.cat([out, out1])
        O = out.reshape(BatchSize, self.n_out, 4, 4)
        return O


class N2(torch.nn.Module):
    def __init__(self, n_in, device):
        super(N2, self).__init__()
        self.n_in = n_in
        self.device = device
        self.secondNet = False
        self.added = 0

        self.register_parameter("aW", torch.nn.Parameter((torch.rand(self.n_in, 1, 1) - 0.5) * 1))
        self.register_parameter("uW", torch.nn.Parameter((torch.rand(self.n_in, 1, 3) - 0.5) * 1))
        self.register_parameter("tW", torch.nn.Parameter((torch.rand(self.n_in, 1, 3) - 0.5) * 1))

        self.register_parameter("eaW", torch.nn.Parameter((torch.rand(1, 1, 1) - 0.5) * 1))
        self.register_parameter("euW", torch.nn.Parameter((torch.rand(1, 1, 3) - 0.5) * 1))
        self.register_parameter("etW", torch.nn.Parameter((torch.rand(1, 1, 3) - 0.5) * 1))

    def forward(self, I):
        BatchSize = I.shape[0]
        aV = self.aW.repeat(BatchSize, 1, 1).reshape(BatchSize, self.n_in, 1)
        tV = self.tW.repeat(BatchSize, 1, 1).reshape(BatchSize, self.n_in, 1, 3)
        saV = None
        stV = None
        eaV = self.eaW.repeat(BatchSize, 1, 1).reshape(BatchSize, 1, 1)
        etV = self.etW.repeat(BatchSize, 1, 1).reshape(BatchSize, 1, 1, 3)
        maskingIndices = torch.tensor([[3, 2, 1], [2, 3, 0], [1, 0, 3]])
        maskingCross = torch.Tensor([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        fixVec = torch.Tensor([[0.0, 0.0, 0.0, 1.0]])
        WM = None

        if self.secondNet:
            saV = self.saW.repeat(BatchSize, 1, 1).reshape(BatchSize, self.added, 1)
            stV = self.stW.repeat(BatchSize, 1, 1).reshape(BatchSize, self.added, 1, 3)

            e2aV = self.e2aW.repeat(BatchSize, 1, 1).reshape(BatchSize, 1, 1)
            e2tV = self.e2tW.repeat(BatchSize, 1, 1).reshape(BatchSize, 1, 1, 3)

        unitstack = torch.Tensor([])
        outerProd = torch.Tensor([])
        for i in range(self.n_in):
            U = torch.sigmoid(self.uW[i])
            U = U / torch.norm(U)
            outerProd = torch.cat([outerProd, torch.ger(U[0], U[0])])
            kunit = torch.cat([U.reshape(1, 3), torch.Tensor([[0.0]])], dim=1)
            unit = kunit[0][maskingIndices] * maskingCross
            unitstack = torch.cat([unitstack, unit])

        unit = unitstack.reshape(self.n_in, 3, 3).repeat(BatchSize, 1, 1, 1)
        outer = outerProd.reshape(self.n_in, 3, 3).repeat(BatchSize, 1, 1, 1)
        rotmat1 = (
            unit * torch.sin(aV.reshape(BatchSize, 1, self.n_in)).reshape(BatchSize, self.n_in, 1, 1)
            + (torch.eye(3).repeat(BatchSize, 1, 1, 1) - outer)
            * torch.cos(aV.reshape(BatchSize, 1, self.n_in)).reshape(BatchSize, self.n_in, 1, 1)
            + outer
        )
        rotmat1 = torch.cat([rotmat1, tV.permute(0, 1, 3, 2)], dim=3)

        if self.secondNet:
            unitstack = torch.Tensor([])
            outerProd = torch.Tensor([])
            for i in range(self.added):
                U = torch.sigmoid(self.suW[i])
                U = U / torch.norm(U)
                outerProd = torch.cat([outerProd, torch.ger(U[0], U[0])])
                kunit = torch.cat([U.reshape(1, 3), torch.Tensor([[0.0]])], dim=1)
                unit = kunit[0][maskingIndices] * maskingCross
                unitstack = torch.cat([unitstack, unit])

            unit = unitstack.reshape(self.added, 3, 3).repeat(BatchSize, 1, 1, 1)
            outer = outerProd.reshape(self.added, 3, 3).repeat(BatchSize, 1, 1, 1)
            rotmat2 = (
                unit * torch.sin(saV.reshape(BatchSize, 1, self.added)).reshape(BatchSize, self.added, 1, 1)
                + (torch.eye(3).repeat(BatchSize, 1, 1, 1) - outer)
                * torch.cos(saV.reshape(BatchSize, 1, self.added)).reshape(BatchSize, self.added, 1, 1)
                + outer
            )
            rotmat2 = torch.cat([rotmat2, stV.permute(0, 1, 3, 2)], dim=3)

            rotmat1 = torch.cat([rotmat1, rotmat2], dim=1)

            unitstack = torch.Tensor([])
            outerProd = torch.Tensor([])
            for i in range(1):
                U = torch.sigmoid(self.e2uW[i])
                U = U / torch.norm(U)
                outerProd = torch.cat([outerProd, torch.ger(U[0], U[0])])
                kunit = torch.cat([U.reshape(1, 3), torch.Tensor([[0.0]])], dim=1)
                unit = kunit[0][maskingIndices] * maskingCross
                unitstack = torch.cat([unitstack, unit])

            unit = unitstack.reshape(1, 3, 3).repeat(BatchSize, 1, 1, 1)
            outer = outerProd.reshape(1, 3, 3).repeat(BatchSize, 1, 1, 1)
            rotmateef = (
                unit * torch.sin(e2aV.reshape(BatchSize, 1, 1)).reshape(BatchSize, 1, 1, 1)
                + (torch.eye(3).repeat(BatchSize, 1, 1, 1) - outer)
                * torch.cos(e2aV.reshape(BatchSize, 1, 1)).reshape(BatchSize, 1, 1, 1)
                + outer
            )

            rotmateef = torch.cat([rotmateef, e2tV.permute(0, 1, 3, 2)], dim=3)
        else:
            unitstack = torch.Tensor([])
            outerProd = torch.Tensor([])
            for i in range(1):
                U = torch.sigmoid(self.euW[i])
                U = U / torch.norm(U)
                outerProd = torch.cat([outerProd, torch.ger(U[0], U[0])])
                kunit = torch.cat([U.reshape(1, 3), torch.Tensor([[0.0]])], dim=1)
                unit = kunit[0][maskingIndices] * maskingCross
                unitstack = torch.cat([unitstack, unit])

            unit = unitstack.reshape(1, 3, 3).repeat(BatchSize, 1, 1, 1)
            outer = outerProd.reshape(1, 3, 3).repeat(BatchSize, 1, 1, 1)
            rotmateef = (
                unit * torch.sin(eaV.reshape(BatchSize, 1, 1)).reshape(BatchSize, 1, 1, 1)
                + (torch.eye(3).repeat(BatchSize, 1, 1, 1) - outer)
                * torch.cos(eaV.reshape(BatchSize, 1, 1)).reshape(BatchSize, 1, 1, 1)
                + outer
            )
            rotmateef = torch.cat([rotmateef, etV.permute(0, 1, 3, 2)], dim=3)

        WM = torch.cat([rotmat1, rotmateef], dim=1)
        WM = torch.cat([WM, fixVec.repeat(BatchSize, self.n_in + self.added + 1, 1, 1)], dim=2)

        # Method 2
        A = torch.eye(4)
        b = torch.tril(torch.ones(self.n_in + self.added + 1, self.n_in + self.added)).flatten()
        c = torch.triu(torch.ones(self.n_in + self.added + 1, self.n_in + self.added), diagonal=1).flatten()

        maskIlower = torch.einsum("ij,k->kij", A, b).reshape(self.n_in + self.added + 1, self.n_in + self.added, 4, 4)
        maskIupper = torch.einsum("ij,k->kij", A, c).reshape(self.n_in + self.added + 1, self.n_in + self.added, 4, 4)

        output1 = torch.matmul(maskIlower, I.reshape(BatchSize, 1, self.n_in + self.added, 4, 4))
        output1 = torch.add(output1, maskIupper)
        output1 = output1.permute(0, 2, 1, 3, 4)

        output = torch.eye(4).repeat(BatchSize, self.n_in + self.added + 1, 1, 1)
        for outnum in range(self.n_in + self.added):
            output = torch.matmul(output, output1[:, outnum])

        out = torch.matmul(output, WM)
        return out

    def secondaryNetwork(self, jointadded):
        self.secondNet = True
        self.added = jointadded
        self.register_parameter("saW", torch.nn.Parameter((torch.rand(self.added, 1, 1) - 0.5) * 1))
        self.register_parameter("suW", torch.nn.Parameter((torch.rand(self.added, 1, 3) - 0.5) * 1))
        self.register_parameter("stW", torch.nn.Parameter((torch.rand(self.added, 1, 3) - 0.5) * 1))

        self.register_parameter("e2aW", torch.nn.Parameter((torch.rand(1, 1, 1) - 0.5) * 1))
        self.register_parameter("e2uW", torch.nn.Parameter((torch.rand(1, 1, 3) - 0.5) * 1))
        self.register_parameter("e2tW", torch.nn.Parameter((torch.rand(1, 1, 3) - 0.5) * 1))


class rotMat(torch.nn.Module):
    def __init__(self, n_in, device):
        super(rotMat, self).__init__()
        self.n_in = n_in
        self.n_out = n_in
        self.device = device
        self.secondNet = False
        self.added = 0

        self.register_parameter("uV", torch.nn.Parameter((torch.rand(self.n_in, 1, 3) - 0.5) * 1))
        self.register_parameter("tV", torch.nn.Parameter((torch.rand(self.n_in, 1, 3) - 0.5) * 1))

    def forward(self, I):
        BatchSize = I.shape[0]
        t = self.tV.repeat(BatchSize, 1, 1).reshape(BatchSize, self.n_in, 1, 3)
        st = None
        maskingIndices = torch.tensor([[3, 2, 1], [2, 3, 0], [1, 0, 3]])
        maskingCross = torch.Tensor([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        fixVec = torch.Tensor([[0.0, 0.0, 0.0, 1.0]])
        out = torch.Tensor([])
        if self.secondNet:
            st = self.stV.repeat(BatchSize, 1, 1).reshape(BatchSize, self.added, 1, 3)

        unitstack = torch.Tensor([])
        outerProd = torch.Tensor([])
        for i in range(self.n_in):
            U = torch.sigmoid(self.uV[i])
            U = U / torch.norm(U)
            outerProd = torch.cat([outerProd, torch.ger(U[0], U[0])])
            kunit = torch.cat([U.reshape(1, 3), torch.Tensor([[0.0]])], dim=1)
            unit = kunit[0][maskingIndices] * maskingCross
            unitstack = torch.cat([unitstack, unit])

        unit1 = unitstack.reshape(self.n_in, 3, 3).repeat(BatchSize, 1, 1, 1)
        outer1 = outerProd.reshape(self.n_in, 3, 3).repeat(BatchSize, 1, 1, 1)

        if self.secondNet:
            unitstack = torch.Tensor([])
            outerProd = torch.Tensor([])
            for i in range(self.added):
                U = torch.sigmoid(self.suV[i])
                U = U / torch.norm(U)
                outerProd = torch.cat([outerProd, torch.ger(U[0], U[0])])
                kunit = torch.cat([U.reshape(1, 3), torch.Tensor([[0.0]])], dim=1)
                unit = kunit[0][maskingIndices] * maskingCross
                unitstack = torch.cat([unitstack, unit])

            unit = unitstack.reshape(self.added, 3, 3).repeat(BatchSize, 1, 1, 1)
            unit1 = torch.cat([unit1, unit], dim=1)
            outer = outerProd.reshape(self.added, 3, 3).repeat(BatchSize, 1, 1, 1)
            outer1 = torch.cat([outer1, outer], dim=1)

            rotmat2 = (
                unit1 * torch.sin(I).reshape(BatchSize, self.n_in + self.added, 1, 1)
                + (torch.eye(3).repeat(BatchSize, 1, 1, 1) - outer1)
                * torch.cos(I).reshape(BatchSize, self.n_in + self.added, 1, 1)
                + outer1
            )
            st = torch.cat([t, st], dim=1)
            rotmat2 = torch.cat([rotmat2, st.permute(0, 1, 3, 2)], dim=3)

            out = torch.cat([rotmat2, fixVec.repeat(BatchSize, self.n_in + self.added, 1, 1)], dim=2)
        else:
            rotmat1 = (
                unit1 * torch.sin(I).reshape(BatchSize, self.n_in, 1, 1)
                + (torch.eye(3).repeat(BatchSize, 1, 1, 1) - outer1) * torch.cos(I).reshape(BatchSize, self.n_in, 1, 1)
                + outer1
            )
            rotmat1 = torch.cat([rotmat1, t.permute(0, 1, 3, 2)], dim=3)
            out = torch.cat([rotmat1, fixVec.repeat(BatchSize, self.n_in, 1, 1)], dim=2)

        return out

    def secondaryNetwork(self, jointadded):
        self.secondNet = True
        self.added = jointadded
        self.register_parameter("suV", torch.nn.Parameter((torch.rand(self.added, 1, 3) - 0.5) * 1))
        self.register_parameter("stV", torch.nn.Parameter((torch.rand(self.added, 1, 3) - 0.5) * 1))


class HTMmodel(torch.nn.Module):
    def __init__(self, n_joint, timeStep):
        super(HTMmodel, self).__init__()
        self.n_joint = n_joint
        self.timeStep = timeStep

        self.rotNet = rotMat(self.n_joint, "cpu")
        self.outNet = N2(self.n_joint, "cpu")

    def forward(self, I):
        BatchSize = I.shape[0]

        newI = I.reshape(BatchSize, 2, self.n_joint) * torch.cat(
            [
                torch.Tensor([self.timeStep]).repeat(1, self.n_joint),
                torch.ones(1, self.n_joint),
            ]
        )
        newI = torch.sum(newI, dim=1)

        rotRes = self.rotNet(newI)
        outRes = self.outNet(rotRes)

        return outRes

    def addSecondary(self, jointadded):
        self.n_joint = self.n_joint + jointadded
        self.rotNet.secondaryNetwork(jointadded)
        self.outNet.secondaryNetwork(jointadded)

    def getAllNets(self):
        return self.rotNet, self.outNet

    def setAllNets(self, nets2, nets3):
        self.rotNet = nets2
        self.outNet = nets3


# Input Output Buffer
class IOElement(BufferElement):
    def __init__(self, p_input: torch.Tensor, p_output: torch.Tensor):

        super().__init__({"input": p_input, "output": p_output})


# Buffer
class MyOwnBuffer(Buffer, torch.utils.data.Dataset):
    def __init__(self, p_size=1):
        Buffer.__init__(self, p_size=p_size)
        self._internal_counter = 0

    def add_element(self, p_elem: BufferElement):
        Buffer.add_element(self, p_elem)
        self._internal_counter += 1

    def get_internal_counter(self):
        return self._internal_counter

    def __getitem__(self,idx):
        return self._data_buffer["input"][idx], self._data_buffer["output"][idx]


class HTMAFctTrans(AdaptiveFunction):
    C_NAME = "HTM Adaptive Function"
    C_BUFFER_CLS = MyOwnBuffer

    def __init__(
        self,
        p_input_space: MSpace,
        p_output_space: MSpace,
        p_output_elem_cls=Element,
        p_threshold=0,
        p_buffer_size=0,
        p_ada=True,
        p_logging=Log.C_LOG_ALL,
        **p_par
    ):

        super().__init__(
            p_input_space=p_input_space,
            p_output_space=p_output_space,
            p_output_elem_cls=p_output_elem_cls,
            p_threshold=p_threshold,
            p_buffer_size=p_buffer_size,
            p_ada=p_ada,
            p_logging=p_logging,
            **p_par
        )

        self.joint_num = p_output_space.get_num_dim() - 6
        self.htm_model = HTMmodel(self.joint_num, 0.01)
        self.optimizer = torch.optim.Adam(self.htm_model.parameters(), lr=3e-4)
        self.loss_dyn = torch.nn.MSELoss()
        self.sim_env = p_par["p_sim_env"]
        self.train_model = True

    def _map(self, p_input: Element, p_output: Element):
        # Prediction
        # Input [Tx, Ty, Tz, Px, Py, Pz, J1, J2, J3, J4, A1, A2, A3, A4]
        # Model Input [A1, A2, A3, A4, J1, J2, J3, J4]
        model_input = deque(p_input.get_values()[6:])
        model_input.rotate(self.joint_num)
        model_input = torch.Tensor([list(model_input)])
        model_output = self.htm_model(model_input)

        # Model Output [HTM1, HTM2, HTM3, HTM4, HTMEE]
        # Output [Tx, Ty, Tz, Px, Py, Pz, J1, J2, J3, J4]

        # Convert HTMEE to [Px, Py, Pz]
        eef = torch.mm(model_output[-1][-1].detach(), torch.Tensor([[0, 0, 0, 1]]).T)

        # Convert [HTM1, HTM2, HTM3, HTM4] to [J1, J2, J3, J4]
        angles = torch.Tensor([])
        thets = torch.zeros(3)
        for idx in range(self.joint_num):
            angle = torch.Tensor(transformations.euler_from_matrix(model_output[-1][idx].detach().numpy())) - thets
            thets = torch.Tensor(transformations.euler_from_matrix(model_output[-1][idx].detach().numpy()))
            angles = torch.cat([angles, torch.norm(angle).reshape(1, 1)], dim=1)

        # Combine Output
        output = p_input.get_values()[:3].copy()
        output.extend(eef[:3, [-1]].cpu().flatten().tolist())
        output.extend(angles.cpu().flatten().tolist())
        p_output.set_values(output)

    def _adapt(self, p_input: Element, p_output: Element) -> bool:
        model_input = deque(p_input.get_values()[6:])
        model_input.rotate(self.joint_num)
        model_input = torch.Tensor([list(model_input)])

        self.sim_env.set_theta(torch.Tensor([p_output.get_values()[6 : 6 + self.joint_num]]))
        self.sim_env.update_joint_coords()

        model_output = self.sim_env.get_homogeneous().reshape(self.joint_num + 1, 4, 4)

        self._add_buffer(IOElement(model_input, model_output))

        if self._buffer.get_internal_counter() % 100 != 0:
            return False

        # Divide Test and Train
        if self.train_model:
            dataset_size = len(self._buffer)
            indices = list(range(dataset_size))
            split = int(np.floor(0.3 * dataset_size))
            np.random.seed(random.randint(1,1000))
            np.random.shuffle(indices)
            train_indices, test_indices = indices[split:], indices[:split]

            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)
            trainer = torch.utils.data.DataLoader(self._buffer, batch_size=100, sampler=train_sampler)
            tester = torch.utils.data.DataLoader(self._buffer, batch_size=100, sampler=test_sampler)

            # Training
            self.htm_model.train()

            for i, (In, Label) in enumerate(trainer):
                outputs = self.htm_model(In)
                loss = self.loss_dyn(outputs, Label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            test_loss = 0

            self.htm_model.eval()
            for i, (In, Label) in enumerate(tester):
                outputs = self.htm_model(In)
                loss = self.loss_dyn(outputs, Label)
                test_loss += loss.item()

            print(test_loss/len(tester))
            if test_loss/len(tester) < 5e-9:
                self.train_model = False

        return True

    def _add_buffer(self, p_buffer_element: IOElement):
        self._buffer.add_element(p_buffer_element)
