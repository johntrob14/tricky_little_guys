from pathlib import Path
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement
from pptx.opc.package import Part
from pptx.opc.packuri import PackURI
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from lxml import etree

ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'outputs/arena'
content=json.loads((O/'main_slides.json').read_text())
prs=Presentation();prs.slide_width=Inches(16);prs.slide_height=Inches(9)
INK='202A36';TEAL='147D79';GRAY='667583'
titles=[
 'Always monitored: the stronger CoT judge\ncatches every recorded leak',
 'Startgate-direct: some leaks evade\neven the stronger CoT judge',
 'Store the leaking implementation\nin the plan, then copy it']
kickers=['LINUXARENA · ALWAYS MONITORED','LINUXARENA · STARTGATE-DIRECT','QWEN3.8-27B OPSD + GRPO · SELECTED TRACE']
footers=[
 'Successful leak + no alert from the named judge, out of 16 attempts. 95% Wilson CIs. CoT view includes assistant text.',
 'Post-start judging. Successful leak + no alert from the named judge, out of 16 attempts. 95% Wilson CIs.',
 'Exact excerpts: opsd1b-sd-pw / episode #13. Selected example; all six judges restricted to post-start turns.']

def text(slide,value,x,y,w,h,size=26,bold=False,color=INK,bullet=False):
 box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
 tf=box.text_frame;tf.word_wrap=True
 tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
 for i,line in enumerate(value.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
  p.text=line;p.font.name='Arial';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color)
  p.space_after=Pt(0);p.space_before=Pt(0);p.line_spacing=1.08
  if bullet:
   pp=p._p.get_or_add_pPr();pp.set('marL',str(Inches(.25)));pp.set('indent',str(-Inches(.25)))
   b=OxmlElement('a:buChar');b.set('char','•');pp.append(b)
 return box

for idx,c in enumerate(content):
 slide=prs.slides.add_slide(prs.slide_layouts[6])
 slide.background.fill.solid();slide.background.fill.fore_color.rgb=RGBColor(255,255,255)
 text(slide,kickers[idx],.55,.22,14.9,.25,13,True,TEAL)
 text(slide,titles[idx],.55,.63,14.9,1.1,34,True)
 image_path=O/c['figure'].replace('.svg','.png')
 picture=slide.shapes.add_picture(str(image_path),Inches(.35),Inches(1.87),width=Inches(15.3),height=Inches(3.825))
 # Native SVG extension with a PNG fallback for PowerPoint/Slides compatibility.
 svgpart=Part(PackURI(f'/ppt/media/arena_figure_{idx+1}.svg'),'image/svg+xml',prs.part.package,(O/c['figure']).read_bytes())
 rid=slide.part.relate_to(svgpart,RT.IMAGE)
 blip=picture._pic.blipFill.blip
 exts=OxmlElement('a:extLst');ext=OxmlElement('a:ext');ext.set('uri','{96DAC541-7B7A-43D3-8B79-37D633B846F1}')
 svg=etree.SubElement(ext,'{http://schemas.microsoft.com/office/drawing/2016/SVG/main}svgBlip',nsmap={'asvg':'http://schemas.microsoft.com/office/drawing/2016/SVG/main'})
 svg.set('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed',rid)
 exts.append(ext);blip.append(exts)
 assert len(c['bullets'])==2
 for j,b in enumerate(c['bullets']):text(slide,b,.72,6.02+j*1.03,14.5,.91,25,bullet=True)
 text(slide,footers[idx],.55,8.44,14.3,.3,12,color=GRAY)
 text(slide,str(idx+1),15.1,8.44,.35,.3,12,color=GRAY)
 notes='PRESENTATION NOTES\n'+c['notes']+'\n\nDISCUSSION POINTS\n'+'\n'.join(f'{j+1}. {s}' for j,s in enumerate(c['discussion_points']))+'\n\nFIGURE CAPTION AND EVIDENCE\n'+c['caption']+'\n\nBACKUP DECK\nhttps://docs.google.com/presentation/d/1PQGuqqYl2TZEAiyupTLwEKbSlpQurqqKWcTJbE7uq2g'
 slide.notes_slide.notes_text_frame.text=notes
prs.core_properties.title='CoT control in LinuxArena: always monitored and startgate-direct'
prs.core_properties.subject='Three main slides, with speaker notes and discussion points'
prs.core_properties.author='John'
dest=O/'linuxarena_main_slides.pptx';prs.save(dest)
check=Presentation(dest)
assert len(check.slides)==3
for slide,c in zip(check.slides,content):
 assert c['notes'] in slide.notes_slide.notes_text_frame.text
 assert all(b in '\n'.join(s.text for s in slide.shapes if s.has_text_frame) for b in c['bullets'])
 assert all(s.left>=0 and s.top>=0 and s.left+s.width<=prs.slide_width and s.top+s.height<=prs.slide_height for s in slide.shapes)
print(dest)
print('Validated three slides, native title/bullet text, SVG plus PNG fallback, speaker notes, and shape bounds.')
