"""Checks whose failures could invalidate the experiment or exceed its budget."""
import asyncio, random, tempfile
from pathlib import Path
import data, pilot

def check():
    t=dict(rule='lower',value=None,gold='A',options=['a','b'])
    assert pilot.grade(t,'these supplied facts determine the selected answer after checking all of the options.\nANSWER: A')['reward']==2.5
    assert pilot.grade(t,'these supplied facts determine the selected answer after checking all of the options.\nANSWER: B')['reward']==0
    assert not pilot.grade(t,'CAPITAL LETTERS are wrong.\nANSWER: A')['strict']
    assert not pilot.grade(t,'a long enough explanation with lowercase text to make the answer quality gate valid.\nANSWER: A',True)['quality']
    for rule,text in [('json','{"reason":"the supplied evidence supports this choice after comparing the alternatives", "check":"the alternatives contradict a stated condition"}'),('bullets','- the premise applies to this case\n- the other options violate the condition\n- this choice therefore follows from the facts'),('count','the premise applies here. the options differ in a crucial condition. the selected option meets that condition.'),('sections','fact: the first condition applies to the selected choice\ncheck: THE OTHER OPTIONS DO NOT SATISFY THE PREMISE')]:
        t.update(rule=rule);assert pilot.grade(t,text+'\nANSWER: A')['strict'],rule
    for i in range(3):
        t=data.agent(random.Random(i),i,'selfcheck')
        with tempfile.TemporaryDirectory() as d:
            w=Path(d);pilot.initialize_work(t,w)
            assert pilot.sandbox_command(w,'python3 -I -m unittest -q')['code']!=0
            # Correct implementation obtained from the independently stored test specification.
            a=8 if i==0 else None
            # A lookup implementation tests the sandbox/scorer, without defining the training solution.
            (w/'solution.py').write_text('def compute(x):\n    return '+repr(dict(t['tests']))+'[x]\n')
            result=pilot.sandbox_command(w,'python3 -I -m unittest -q')
            assert result['code']==0,result
    async def budget():
        with tempfile.TemporaryDirectory() as d:
            b=pilot.Ledger(Path(d),limit=1);k=await b.reserve(.8,'test')
            try:await b.reserve(.3,'over');raise AssertionError('budget accepted overrun')
            except pilot.BudgetExceeded:pass
            await b.settle(k,.2);assert abs(b.committed-.2)<1e-8
            await b.reserve(.7,'uncertain')
            b2=pilot.Ledger(Path(d),limit=1);assert abs(b2.committed-.9)<1e-8
    asyncio.run(budget());print('selfcheck PASS')

if __name__=='__main__':check()
