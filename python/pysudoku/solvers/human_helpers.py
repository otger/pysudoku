from pysudoku.common import SudokuMatrix, CandidatesMatrix

# https://www.sudokuoftheday.com/about/difficulty/
# https://www.sudokuoftheday.com/techniques/
class StepType:
    SingleCandidate = 'scd'
    SinglePosition = 'spo'
    CandidateLines = 'cdl'
    DoublePairs = 'dbp'
    MultipleLines = 'mtl'
    NakedPair = 'nkp'
    HiddenPair = 'hdp'
    NakedTriple = 'nkt'
    HiddenTriple = 'hdt'
    XWing = 'xwg'
    ForcingChains = 'fcc'
    NakedQuad = 'nkq'
    HiddenQuad = 'hdq'
    Swordfish = 'swf'
    Guessing = 'gsg'
    CleanCandidates = 'cct'
    Unknown = 'unk'
    ScanCandidates = 'scan'
    InitState = 'init'


StepsCosts = {
    StepType.SingleCandidate: {'first': 100, 'seconds': 100},
    StepType.SinglePosition: {'first': 100, 'seconds': 100},
    StepType.CandidateLines: {'first': 350, 'seconds': 200},
    StepType.DoublePairs: {'first': 500, 'seconds': 250},
    StepType.MultipleLines: {'first': 700, 'seconds': 400},
    StepType.NakedPair: {'first': 750, 'seconds': 500},
    StepType.HiddenPair: {'first': 1500, 'seconds': 1200},
    StepType.NakedTriple: {'first': 2000, 'seconds': 1400},
    StepType.HiddenTriple: {'first': 2400, 'seconds': 1600},
    StepType.XWing: {'first': 2800, 'seconds': 1600},
    StepType.ForcingChains: {'first': 4200, 'seconds': 2100},
    StepType.NakedQuad: {'first': 5000, 'seconds': 4000},
    StepType.HiddenQuad: {'first': 7000, 'seconds': 5000},
    StepType.Swordfish: {'first': 8000, 'seconds': 6000},
    StepType.Guessing: {'first': 9000, 'seconds': 10000},
    StepType.CleanCandidates: {'first': 0, 'seconds': 0},
    StepType.Unknown: {'first': 0, 'seconds': 0},
    StepType.ScanCandidates: {'first': 0, 'seconds': 0},
    StepType.InitState: {'first': 0, 'seconds': 0},
}


class StepResolution:
    all = 0
    mid = 5
    none = 100


class Step:
    def __init__(self, solutions_str, candidates_str, step_type: str, msg='', prev_step=None,
                 index=0):
        self.solutions = SudokuMatrix(solutions_str)
        self.candidates = CandidatesMatrix.from_str(candidates_str)
        self.step_type = step_type
        self.msg = msg
        self.prev = prev_step
        self.acc_types = [self.step_type]
        if prev_step:
            prev_step.next = self
            self.acc_types.extend(prev_step.acc_types)
        self.next = None
        self.index = index
        self.symbols = [a for a in list(set(self.solutions.v.flatten())) if a != '-']
        self.fill_symbols()

    @property
    def cost(self):
        _repeat = 'first'
        if self.acc_types.count(self.step_type) > 1:
            _repeat = 'seconds'
        return StepsCosts[self.step_type][_repeat]

    def fill_symbols(self):
        if len(self.symbols) < 9:
            for i in range(9):
                if str(i) not in self.symbols:
                    self.symbols.append(str(i))
                if len(self.symbols) == 9:
                    break

    def is_broken(self):
        for i, row in enumerate(self.solutions.v):
            for j, cell in enumerate(row):
                if cell == '-':
                    if len(self.candidates.v[i, j]) == 0:
                        return i, j
        ans = self.candidates.is_broken()
        if ans:
            return ans
        return False

    def solved(self) -> bool:
        if '-' in self.solutions.v.flatten():
            return False
        s = set(self.symbols)
        for r in range(9):
            if set(self.solutions.get_row(r)) != s:
                return False
        for c in range(9):
            if set(self.solutions.get_column(c)) != s:
                return False
        return True
