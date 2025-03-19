import os
import re

import matplotlib.pyplot as plt
import numpy as np
from hip_research.utils import setup_seaborn
from rouge_score import rouge_scorer

GOVREPORT_TRUTH = "Multiyear procurement (MYP) and block buy contracting (BBC) are special contracting mechanisms that Congress permits the Department of Defense (DOD) to use for a limited number of defense acquisition programs. Compared to the standard or default approach of annual contracting, MYP and BBC have the potential for reducing weapon procurement costs by a few or several percent. Under annual contracting, DOD uses one or more contracts for each year's worth of procurement of a given kind of item. Under MYP, DOD instead uses a single contract for two to five years' worth of procurement of a given kind of item without having to exercise a contract option for each year after the first year. DOD needs congressional approval for each use of MYP. There is a permanent statute governing MYP contracting—10 U.S.C. 2306b. Under this statute, a program must meet several criteria to qualify for MYP. Compared with estimated costs under annual contracting, estimated savings for programs being proposed for MYP have ranged from less than 5% to more than 15%, depending on the particulars of the program in question, with many estimates falling in the range of 5% to 10%. In practice, actual savings from using MYP rather than annual contracting can be difficult to observe or verify because of cost growth during the execution of the contract due to changes in the program independent of the use of MYP rather than annual contracting. BBC is similar to MYP in that it permits DOD to use a single contract for more than one year's worth of procurement of a given kind of item without having to exercise a contract option for each year after the first year. BBC is also similar to MYP in that DOD needs congressional approval for each use of BBC. BBC differs from MYP in the following ways: There is no permanent statute governing the use of BBC. There is no requirement that BBC be approved in both a DOD appropriations act and an act other than a DOD appropriations act. Programs being considered for BBC do not need to meet any legal criteria to qualify for BBC, because there is no permanent statute governing the use of BBC that establishes such criteria. A BBC contract can cover more than five years of planned procurements. Economic order quantity (EOQ) authority—the authority to bring forward selected key components of the items to be procured under the contract and purchase the components in batch form during the first year or two of the contract—does not come automatically as part of BBC authority because there is no permanent statute governing the use of BBC that includes EOQ authority as an automatic feature. BBC contracts are less likely to include cancellation penalties. Potential issues for Congress concerning MYP and BBC include whether to use MYP and BBC in the future more frequently, less frequently, or about as frequently as they are currently used; whether to create a permanent statute to govern the use of BBC, analogous to the permanent statute that governs the use of MYP; and whether the Coast Guard should begin making use of MYP and BBC."
GOVREPORT_SLLM = "The report analyzes the implementation of multiyear procurements (MYP) in the Department of Defense (DOD) since 1990, focusing on the FY2011-FY2021 period. It highlights the evolution of MYP policies, the role of the Under Secretary of Defense for Acquisition, Logistics, and Technology, and the impact of MYP on procurement processes. The report finds that MYP has grown significantly, with a total value of $145 billion in FY2021, representing 40% of DOD acquisitions. It identifies key factors contributing to MYP's success, such as improved planning, reduced administrative burden, and enhanced competition. However, challenges remain, including the need for better data management, increased transparency, and the potential for unintended consequences. The report also examines the impact of MYP on specific programs, with a focus on major systems, and provides a list of programs approved for MYP in annual DOD appropriations acts since FY1990. Overall, the study underscores the importance of MYP in streamlining defense procurement while recognizing the need for ongoing evaluation and improvement."
GOVREPORT_HIP = "This report provides an overview of multiyear procurement (MYP) and block buy contracting (BBC) mechanisms employed by the Department of Defense (DOD) for defense acquisition programs. MYP allows for procurement of items over multiple years, typically two to five, reducing annual contract requirements. BBC is a similar approach, but with less formal statutory governance. The report highlights that MYP has been used for a limited number of programs, with potential savings ranging from 5% to 15% compared to annual contracting.\n\nThe use of MYP and BBC impacts defense funding, procurement flexibility, and the industrial base. A permanent statute, 10 U.S.C. 2306b, governs MYP, while BBC operates on a case-by-case basis. The report suggests that Congress may consider using these mechanisms more or less frequently, or creating a similar statute for BBC.\n\nThe Coast Guard, though part of the Department of Defense, has its own contracting regulations for block buy contracting, with the authority to use BBC for certain major acquisition programs. The report highlights the potential cost savings for the Coast Guard if MYP or BBC were applied to its major programs, such as the Offshore Patrol Cutter and polar icebreaker programs.\n\nIn summary, the report discusses the benefits and potential risks of MYP and BBC, their current usage, and the possibility of expanding their application or creating a permanent statute for block buy contracting. The report also underscores the need to balance cost savings with the flexibility to adjust procurement programs in response to strategic or budgetary changes.\n"

MULTINEWS_TRUTH = 'Under circumstances that sound straight out of a Le Carre novel, Mohamed Morsi has met with an outsider for the first time since being dumped by Egypt\'s military, holding a midnight meeting with the European Union\'s top diplomat at an "undisclosed location" last night, the Wall Street Journal. Though many have feared for Morsi\'s well-being after almost a month in captivity, Catherine Ashton, the EU\'s foreign affairs and security policy chief, reports that he is well and has access to TV and newspapers, Reuters reports. The pair had a "very long and in-depth conversation," Ashton says. Ashton, who has met with several other Egyptian political figures while in the country, says she wasn\'t there to help Morsi break out, but rather to facilitate discussions toward a peaceful end to the country\'s political unrest, the Washington Post reports. "First of all, we are here to help. We are not here to impose," she says. "The people of Egypt will determine their own future." But will Morsi play ball? "He\'s a very proud man," says a source close to the former leader, per the Journal. "I think the idea of doing anything on (the) army\'s terms will not work with him."'
MULTINEWS_SLLM = "The summary:\n\nIn recent news, European Union High Representative for Foreign Affairs and Security Policy, Federica Mogherini, visited Egypt to discuss the ongoing political crisis following the ousting of President Mohamed Morsi. Mogherini met with various Egyptian political figures, including General Abdel Fatah al-Sissi, the head of the armed forces, and Vice President Mohamed ElBaradei. Her visit aimed to facilitate dialogue towards a political solution and promote democracy in the country.\n\nDuring her meeting with Morsi, Mogherini expressed hope for Egypt's future and emphasized that the E.U. was there to assist, not impose solutions. She noted that Egyptians should determine their own path, but also stressed the importance of a democratic transition. Morsi's supporters, represented by the Freedom and Justice Party, insisted on \"the return of the president\" as a condition for any political settlement, referring to his reinstatement to power.\n\nThe visit came amidst ongoing protests and violence, with at least 80 pro-Morsi demonstrators and a police officer killed over the weekend. Mogherini's visit seemed to temporarily calm tensions in Cairo, but demonstrations continued, with protesters marching towards the military intelligence headquarters.\n\nU.S. Defense Secretary Chuck Hagel also weighed in, urging restraint in handling protesters and calling for an inclusive reconciliation process. The situation remains complex, with the Muslim Brotherhood's demand for constitutional legitimacy, which they interpret as Morsi's restoration to power, a central point of contention.\n\nOverall, Mogherini's visit highlights the international community's interest in Egypt's stability and the need for a peaceful, democratic resolution to the crisis."
MULTINEWS_HIP = "In a significant development, European Union foreign policy chief Catherine Ashton met with deposed Egyptian President Mohamed Morsi on Monday night, marking his first contact with an independent official since his ousting nearly a month ago. Ashton, who is mediating efforts to resolve Egypt's political crisis, confirmed Morsi's well-being, stating that he had access to television and newspapers during their two-hour conversation. The meeting took place at an undisclosed location, and while Ashton did not disclose details, she expressed hope for a political solution.\n\nMorsi's supporters have been protesting, with clashes between security forces and demonstrators resulting in at least 80 deaths over the weekend. The former president is under investigation for espionage and murder, allegations his supporters view as politically motivated. Ashton also met with interim Vice President Mohamed ElBaradei, who expressed the need for the Muslim Brotherhood to be part of the country's political roadmap but without returning Morsi to power.\n\nAshton's visit, which included meetings with various political figures, Gen. Abdel Fattah al-Sissi, and representatives of the Muslim Brotherhood, aimed to facilitate discussions towards a democratic resolution. However, the Brotherhood's Freedom and Justice Party insisted that any solution must involve Morsi's return, a demand that aligns with the concept of \"constitutional legitimacy\" for their supporters.\n\nThe US Defense Secretary, Chuck Hagel, also weighed in, urging restraint and calling for an inclusive reconciliation process to address the ongoing crisis in Egypt. Ashton's visit appears to have temporarily calmed tensions in the capital but tensions remain high, with pro-Morsi demonstrations continuing."


def print_scores(name, scores):
    print(
        f'[{name}] ROUGE-1: {scores["rouge1"].fmeasure * 100:.2f}, ROUGE-2: {scores["rouge2"].fmeasure * 100:.2f}, ROUGE-L: {scores["rougeL"].fmeasure * 100:.2f}'
    )


scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

scores = scorer.score(GOVREPORT_TRUTH, GOVREPORT_HIP)
print_scores("gov_report, hip ", scores)
scores = scorer.score(GOVREPORT_TRUTH, GOVREPORT_SLLM)
print_scores("gov_report, sllm", scores)

scores = scorer.score(MULTINEWS_TRUTH, MULTINEWS_HIP)
print_scores("multi_news, hip ", scores)
scores = scorer.score(MULTINEWS_TRUTH, MULTINEWS_SLLM)
print_scores("multi_news, sllm", scores)

GOVREPORT_CONTENT = """
This report provides background information and issues for Congress on multiyear procurement (MYP) and block buy contracting (BBC),
which are special contracting mechanisms that Congress permits the Department of Defense (DOD) to use for a limited number of defense
acquisition programs. Compared to the standard or default approach of annual contracting, MYP and BBC have the potential for reducing
weapon procurement costs by a few or several percent. Potential issues for Congress concerning MYP and BBC include whether to use MYP
and BBC in the future more frequently, less frequently, or about as frequently as they are currently used; whether to create a permanent
statute to govern the use of BBC, analogous to the permanent statute that governs the use of MYP; and whether the Coast Guard should begin
making use of MYP and BBC. Congress's decisions on these issues could affect defense acquisition practices, defense funding requirements,
and the defense industrial base. A contract that the Air Force has for the procurement of Evolved Expendable Launch Vehicle (EELV)
Launch Services (ELS) has sometimes been referred to as a block buy, but it is not an example of block buy contracting as discussed in
this report. The Air Force in this instance is using the term block buy to mean something different. This report does not discuss the
ELS contract. (For additional discussion, see " Terminology Alert: Block Buy Contracting vs. Block Buys " below.) In discussing MYP,
BBC, and incremental funding, it can be helpful to distinguish contracting mechanisms from funding approaches. The two are often mixed
together in discussions of DOD acquisition, sometimes leading to confusion. Stated briefly Funding approaches are ways that Congress
can appropriate funding for weapon procurement programs, so that DOD can then put them under contract. Examples of funding approaches
include traditional full funding (the standard or default approach), incremental funding, and advance appropriations. Any of these funding
approaches might make use of advance procurement (AP) funding. Contracting mechanisms are ways for DOD to contract for the procurement of
weapons systems, once funding for those systems has been appropriated by Congress. Examples of contracting mechanisms include annual
contracting (the standard or default DOD approach), MYP, and BBC. Contracting mechanisms can materially change the total procurement
cost of a ship. The use of a particular funding approach in a defense acquisition program does not dictate the use of a particular
contracting mechanism. Defense acquisition programs consequently can be implemented using various combinations of funding approaches
and contracting mechanisms. Most DOD weapon acquisition programs use a combination of traditional full funding and annual contracting.
A few programs, particularly certain Navy shipbuilding programs, use incremental funding as their funding approach. A limited number of DOD
programs use MYP as their contracting approach, and to date three Navy shipbuilding programs have used BBC as their contracting approach.
The situation is summarized in Table 1 . This report focuses on the contracting approaches of MYP and BBC and how they compare to annual
contracting. Other CRS reports discuss the funding approaches of traditional full funding, incremental funding, and advance appropriations.
What is MYP, and how does it differ from annual contracting? MYP, also known as multiyear contracting, is an alternative to the standard
or default DOD approach of annual contracting. Under annual contracting, DOD uses one or more contracts for each year's worth of
procurement of a given kind of item. Under MYP, DOD instead uses a single contract for two to five years' worth of procurement of a
given kind of item, without having to exercise a contract option for each year after the first year. DOD needs congressional approval
for each use of MYP. To illustrate the basic difference between MYP and annual contracting, consider a hypothetical DOD program to procure
20 single-engine aircraft of a certain kind over the 5-year period FY2018-FY2022, at a rate of 4 aircraft per year: Under annual contracting ,
DOD would issue one or more contracts for each year's procurement of four aircraft. After Congress funds the procurement of the first
four aircraft in FY2018, DOD would issue one or more contracts (or exercise a contract option) for those four aircraft. The next year,
after Congress funds the procurement of the next four aircraft in FY2019, DOD would issue one or more contracts (or exercise a contract
option) for those four aircraft, and so on. Under MYP , DOD would issue one contract covering all 20 aircraft to be procured during the
5-year period FY2018-FY2022. DOD would award this contract in FY2018, at the beginning of the five-year period, following congressional
approval to use MYP for the program, and congressional appropriation of the FY2018 funding for the program. To continue the implementation
of the contract over the next four years, DOD would request the FY2019 funding for the program as part of DOD's proposed FY2019 budget,
the FY2020 funding as part of DOD's proposed FY2020 budget, and so on. How much can MYP save? Compared with estimated costs under annual
contracting, estimated savings for programs being proposed for MYP have ranged from less than 5% to more than 15%, depending on the
particulars of the program in question, with many estimates falling in the range of 5% to 10%. In practice, actual savings from using
MYP rather than annual contracting can be difficult to observe or verify because of cost growth during the execution of the contract
that was caused by developments independent of the use of MYP rather than annual contracting. A February 2012 briefing by the Cost
Assessment and Program Evaluation (CAPE) office within the Office of the Secretary of Defense (OSD) states that "MYP savings analysis
is difficult due to the lack of actual costs on the alternative acquisition path, i.e., the path not taken." The briefing states that
CAPE up to that point had assessed MYP savings for four aircraft procurement programs—F/A-18E/F strike fighters, H-60 helicopters,
V-22 tilt-rotor aircraft, and CH-47F helicopters—and that CAPE's assessed savings ranged from 2% to 8%. A 2008 Government Accountability
Office (GAO) report stated that DOD does not have a formal mechanism for tracking multiyear results against original expectations and
makes few efforts to validate whether actual savings were achieved by multiyear procurement. It does not maintain comprehensive central
records and historical information that could be used to enhance oversight and knowledge about multiyear performance to inform and
improve future multiyear procurement (MYP) candidates. DOD and defense research centers officials said it is difficult to assess
results because of the lack of historical information on multiyear contracts, comparable annual costs, and the dynamic acquisition environment.
How does MYP potentially save money? Compared to annual contracting, using MYP can in principle reduce the cost of the weapons being procured
in two primary ways: Contractor optimization of workforce and production facilities . An MYP contract gives the contractor (e.g., an airplane
manufacturer or shipbuilder) confidence that a multiyear stream of business of a known volume will very likely materialize. This confidence
can permit the contractor to make investments in the firm's workforce and production facilities that are intended to optimize the facility
for the production of the items being procured under the contract. Such investments can include payments for retaining or training workers,
or for building, expanding, or modernizing production facilities. Under annual contracting, the manufacturer might not have enough confidence
about its future stream of business to make these kinds of investments, or might be unable to convince its parent firm to finance them. E conomic
order quan tity (EOQ) purchases of selected long-leadtime components. Under an MYP contract, DOD is permitted to bring forward selected key
components of the items to be procured under the contract and to purchase the components in batch form during the first year or two of the
contract. In the hypothetical example introduced earlier, using MYP could permit DOD to purchase, say, the 20 engines for the 20 aircraft in
the first year or two of the 5-year contract. Procuring selected components in this manner under an MYP contract is called an economic
order quantity (EOQ) purchase. EOQ purchases can reduce the procurement cost of the weapons being procured under the MYP contract by allowing
the manufacturers of components to take maximum advantage of production economies of scale that are possible with batch orders. What gives the
contractor confidence that the multiyear stream of business will materialize? At least two things give the contractor confidence that DOD will
not terminate an MYP contract and that the multiyear stream of business consequently will materialize: For a program to qualify for MYP, DOD
must certify, among other things, that the minimum need for the items to be purchased is expected to remain substantially unchanged during
the contract in terms of production rate, procurement rate, and total quantities. Perhaps more important to the contractor, MYP contracts
include a cancellation penalty intended to reimburse a contractor for costs that the contractor has incurred (i.e., investments the contractor has made)
in anticipation of the work covered under the MYP contract. The undesirability of paying a cancellation penalty acts as a disincentive for
the government against canceling the contract. (And if the contract is canceled, the cancellation penalty helps to make the contractor whole.)
Is there a permanent statute governing MYP contracting? There is a permanent statute governing MYP contracting—10 U.S.C. 2306b. The statute was
created by Section 909 of the FY1982 Department of Defense Authorization Act ( S. 815 / P.L. 97-86 of December 1, 1981), revised and reorganized
by Section 1022 of the Federal Acquisition Streamlining Act of 1994 ( S. 1587 / P.L. 103-355 of October 13, 1994), and further amended on several
occasions since. For the text of 10 U.S.C. 2306b, see Appendix A . DOD's use of MYP contracting is further governed by DOD acquisition regulations.
Under this statute, what criteria must a program meet to qualify for MYP? 10 U.S.C. 2306b(a) states that to qualify for MYP, a program must
meet several criteria, including the following: Significant savings. DOD must estimate that using an MYP contract would result in "significant
savings" compared with using annual contracting. Realistic cost estimates . DOD's estimates of the cost of the MYP contract and the anticipated
savings must be realistic. Stable need for the items. DOD must expect that its minimum need for the items will remain substantially unchanged
during the contract in terms of production rate, procurement rate, and total quantities. Stable design for the items . The design for the items
to be acquired must be stable, and the technical risks associated with the items must not be excessive. 10 U.S.C. includes provisions requiring
the Secretary of Defense or certain other DOD officials to find, determine, or certify that these and other statutory requirements for using
MYP contracts have been met, and provisions requiring the heads of DOD agencies to provide written notifications of certain things to the
congressional defense committees 30 days before awarding or initiating an MYP contract, or 10 days before terminating one. 10 U.S.C. 2306b
also requires DOD MYP contracts to be fixed-price type contracts. What is meant by " significant savings"? The amount of savings required under
10 U.S.C. 2306b to qualify for using an MYP contract has changed over time; the requirement was changed from "substantial savings" to "significant savings"
by Section 811 of the FY2016 National Defense Authorization Act ( S. 1356 / P.L. 114-92 of November 25, 2015). The joint explanatory statement
for the FY2016 National Defense Authorization Act states the following regarding Section 811: Amendment relating to multiyear contract authority for
acquisition of property (sec. 811) The House bill contained a provision (sec. 806) that would strike the existing requirement that the head of an
agency must determine that substantial savings would be achieved before entering into a multiyear contract. The Senate amendment contained no similar provision.
The Senate recedes with an amendment that would require that significant savings would be achieved before entering into a multiyear contract.
The conferees agree that the government should seek to maximize savings whenever it pursues multiyear procurement. However, the conferees also
agree that significant savings (estimated to be greater than $250.0 million), and other benefits, may be achieved even if it does not equate to a minimum of
10 percent savings over the cost of an annual contract. The conferees expect a request for authority to enter into a multiyear contract will include (1)
the estimated cost savings, (2) the minimum quantity needed, (3) confirmation that the design is stable and the technical risks are not excessive, and (4)
any other rationale for entering into such a contract. In addition, 10 U.S.C. 2306b states the following: If for any fiscal year a multiyear contract to be
entered into under this section is authorized by law for a particular procurement program and that authorization is subject to certain conditions established
by law (including a condition as to cost savings to be achieved under the multiyear contract in comparison to specified other contracts) and if it appears
(after negotiations with contractors) that such savings cannot be achieved, but that significant savings could nevertheless be achieved through the use of
a multiyear contract rather than specified other contracts, the President may submit to Congress a request for relief from the specified cost savings that
must be achieved through multiyear contracting for that program. Any such request by the President shall include details about the request for a multiyear
contract, including details about the negotiated contract terms and conditions. What is meant by "stable design"? The term "stable design" is generally
understood to mean that the design for the items to be procured is not expected to change substantially during the period of the contract. Having a stable
design is generally demonstrated by having already built at least a few items to that design (or in the case of a shipbuilding program, at least one ship
to that design) and concluding, through testing and operation of those items, that the design does not require any substantial changes during the period
of the contract. What happens if Congress does not provide the annual funding requested by DOD to continue the implementation of the contract? If Congress
does not provide the funding requested by DOD to continue the implementation of an MYP contract, DOD would be required to renegotiate, suspend, or terminate
the contract. Terminating the contract could require the government to pay a cancellation penalty to the contractor. Renegotiating or suspending the contract
could also have a financial impact. What effect does using MYP have on flexibility for making procurement changes? A principal potential disadvantage of
using MYP is that it can reduce Congress's and DOD's flexibility for making changes (especially reductions) in procurement programs in future years in
response to changing strategic or budgetary circumstances, at least without incurring cancellation penalties. In general, the greater the portion of DOD's
procurement account that is executed under MYP contracts, the greater the potential loss of flexibility. The use of MYP for executing some portion of the DOD
procurement account means that if policymakers in future years decide to reduce procurement spending below previously planned levels, the spending reduction
might fall more heavily on procurement programs that do not use MYP, which in turn might result in a less-than-optimally balanced DOD procurement effort. How
does Congress approve the use of MYP? Congress approves the use of MYP on a case-by-case basis, typically in response to requests by DOD. Congressional app
roval for DOD MYP contracts with a value of more than $500 million must occur in two places: an annual DOD appropriations act and an act other than the ann
ual DOD appropriations act. In annual DOD appropriations acts, the provision permitting the use of MYP for one or more defense acquisition programs is typic
ally included in the title containing general provisions, which typically is Title VIII. As shown in Table B-2 , since FY2011, it has been Section 8010. An
annual national defense authorization act (NDAA) is usually the act other than an a
ppropriations act in which provisions granting authority for using MYP contracting on individual defense acquisition programs are included. Such provisions
typically occur in Title I of the NDAA, the title covering procurement programs. Provisions in which Congress approves the use of MYP for a particular defens
e acquisition program may include specific conditions for that program in addition to the requirements and conditions of 10 U.S.C. 2306b. How often is MYP us
ed? MYP is used for a limited number of DOD acquisition programs. As shown in the Appendix B , annual DOD appropriations acts since FY1990 typically have ap
proved the use of MYP for zero to a few DOD programs each year. An August 28, 2017, press report states the following: The Pentagon's portfolio of active mu
ltiyear procurement contracts is on track to taper from $10.7 billion in fiscal year 2017—or more than 8 percent of DOD procurement spending—to $1.2 billion
by FY-19, according to data recently compiled by the Pentagon comptroller for lawmakers. However, there are potential new block-buy deals in the works, incl
uding several large Navy deals. According to the Multiyear Procurement Contracts Report for FY-17, which includes data current as of June 27, seven major de
fense acquisition programs are being purchased through multiyear procurement contracts, collectively obligating the U.S. government to spend $16 billion acr
oss the five-year spending plan with $14.5 billion of the commitments lashed to FY-17 and FY-18. In an interview published on January 13, 2014, Sean Stackle
y, the Assistant Secretary of the Navy for Research, Development, and Acquisition (i.e., the Navy's acquisition executive), stated the following: What the i
ndustrial base clamors for is stability, so they can plan, invest, train their work force. It gives them the ability in working with say, the Street [Wall S
treet], to better predict their own performance, then meet expectations in the same fashion we try to meet our expectations with the Hill. It's emblematic o
f stability that we've got more multiyear programs in the Department of the Navy than the rest of the Department of Defense combined. We've been able to har
vest from that significant savings, and that has been key to solving some of our budget problems. It's allowed us in certain cases to put the savings right
back into other programs tied to requirements. A February 2012 briefing by the Cost Assessment and Program Evaluation (CAPE) office within the Office of the
Secretary of Defense (OSD) shows that the total dollar value of DOD MYP contracts has remained more or less stable between FY2000 and FY2012 at roughly $7 b
illion to $13 billion per year. The briefing shows that since the total size of DOD's procurement budget has increased during this period, the portion of DO
D's total procurement budget accounted for by programs using MYP contracts has declined from about 17% in FY2000 to less than 8% in FY2012. The briefing als
o shows that the Navy makes more use of MYP contracts than does the Army or Air Force, and that the Air Force made very little use of MYP in FY2010-FY2012.
A 2008 Government Accountability Office (GAO) report stated the following: Although DOD had been entering into multiyear contracts on a limited basis prior
to the 1980s, the Department of Defense Authorization Act, [for fiscal year] 1982, codified the authority for DOD to procure on a multiyear basis major weap
on systems that meet certain criteria. Since that time, DOD has annually submitted various weapon systems as multiyear procurement candidates for congressio
nal authorization. Over the past 25 years, Congress has authorized the use of multiyear procurement for approximately 140 acquisition programs, including so
me systems approved more than once. What is BBC, and how does it compare to MYP? BBC is similar to MYP in that it permits DOD to use a single contract for m
ore than one year's worth of procurement of a given kind of item without having to exercise a contract option for each year after the first year. BBC is als
o similar to MYP in that DOD needs congressional approval for each use of BBC. BBC differs from MYP in the following ways: There is no permanent statute gov
erning the use of BBC. There is no requirement that BBC be approved in both a DOD appropriations act and an act other than a DOD appropriations act. Program
s being considered for BBC do not need to meet any legal criteria to qualify for BBC because there is no permanent statute governing the use of BBC that est
ablishes such criteria. A BBC contract can cover more than five years of planned procurements. The BBC contracts that were used by the Navy for procuring Li
ttoral Combat Ships (LCSs), for example, covered a period of seven years (FY2010-FY2016). Economic order quantity (EOQ) authority does not come automaticall
y as part of BBC authority because there is no permanent statute governing the use of BBC that includes EOQ authority as an automatic feature. To provide EO
Q authority as part of a BBC contract, the provision granting authority for using BBC in a program may need to state explicitly that the authority to use BB
C includes the authority to use EOQ. BBC contracts are less likely to include cancellation penalties. Given the one key similarity between BBC and MYP (the
use of a single contract for more than one year's worth of procurement), and the various differences between BBC and MYP, BBC might be thought of as a less
formal stepchild of MYP. When and why was BBC invented? BBC was invented by Section 121(b) of the FY1998 National Defense Authorization Act ( H.R. 1119 / P.
L. 105-85 of November 18, 1997), which granted the Navy the authority to use a single contract for the procurement of the first four Virginia (SSN-774) clas
s attack submarines. The 4 boats were scheduled to be procured during the 5-year period FY1998-FY2002 in annual quantities of 1-1-0-1-1. Congress provided t
he authority granted in Section 121(b) at least in part to reduce the combined procurement cost of the four submarines. Using MYP was not an option for the
Virginia-class program at that time because the Navy had not even begun, let alone finished, construction of the first Virginia-class submarine, and consequ
ently could not demonstrate that it had a stable design for the program. When Section 121(b) was enacted, there was no name for the contracting authority it
provided. The term block buy contracting came into use later, when observers needed a term to refer to the kind of contracting authority that Congress autho
rized in Section 121(b). As discussed in the next section, this can cause confusion, because the term block buy was already being used in discussions of DOD
acquisition to refer to something else. What's the difference between block buy cont r acting and block buys? In discussions of defense procurement, the ter
m "block buy" by itself (without "contracting" at the end) is sometimes used to refer to something quite different from block buy contracting—namely, the si
mple act of funding the procurement of more than one copy of an item in a single year, particularly when no more than one item of that kind might normally b
e funded in a single year. For example, when Congress funded the procurement of two aircraft carriers in FY1983, and another two in FY1988, these acts were
each referred to as block buys, because aircraft carriers are normally procured one at a time, several years apart from one another. This alternate meaning
of the term block buy predates by many years the emergence of the term block buy contracting. The term block buy is still used in this alternate manner, whi
ch can lead to confusion in discussions of defense procurement. For example, for FY2017, the Air Force requested funding for procuring five Evolved Expendab
le Launch Vehicles (EELVs) for its EELV Launch Services (ELS) program. At the same time, Navy officials sometimes refer to the use of block buy contracts fo
r the first four Virginia-class submarines, and in the LCS program, as block buys, when they might be more specifically referred to as instances of block bu
y contract ing . How much can BBC save, compared with MYP? BBC can reduce the unit procurement costs of ships by amounts less than or perhaps comparable to
those of MYP, if the authority granted for using BBC explicitly includes authority for making economic order quantity (EOQ) purchases of components. If the
authority granted for using BBC does not explicitly include authority for making EOQ purchases, then the savings from BBC will be less. Potential savings un
der BBC might also be less than those under MYP if the BBC contract does not include a cancellation penalty, or includes one that is more limited than typic
ally found in an MYP contract, because this might give the contractor less confidence than would be the case under an MYP contract that the future stream of
business will materialize as planned, which in turn might reduce the amount of money the contractor invests to optimize its workforce and production facilit
ies for producing the items to be procured under the contract. How frequently has BBC been used? Since its use at the start of the Virginia-class program, B
BC has been used very rarely. The Navy did not use it again in a shipbuilding program until December 2010, when it awarded two block buy contracts, each cov
ering 10 LCSs to be procured over the six-year period FY2010-FY2015, to the two LCS builders. (Each contract was later amended to include an 11 th ship in F
Y2016, making for a total of 22 ships under the two contracts.) A third example is the John Lewis (TAO-205) class oiler program, in which the Navy is using
a block buy contract to procure the first six ships in the program. A fourth example, arguably, is the Air Force's KC-46 aerial refueling tanker program, wh
ich is employing a fixed price incentive fee (FPIF) development contract that includes a "back end" commitment to procure certain minimum numbers of KC-46s
in certain fiscal years. When might BBC be suitable as an alternative to MYP? BBC might be particularly suitable as an alternative to MYP in cases where usi
ng a multiyear contract can reduce costs, but the program in question cannot meet all the statutory criteria needed to qualify for MYP. As shown in the case
of the first four Virginia-class boats, this can occur at or near the start of a procurement program, when design stability has not been demonstrated throug
h the production of at least a few of the items to be procured (or, for a shipbuilding program, at least one ship). What i s the difference between an MYP o
r block buy contract and a contract with options? The military services sometimes use contracts with options to procure multiple copies of an item that are
procured over a period of several years. The Navy, for example, used a contract with options to procure Lewis and Clark (TAKE-1) class dry cargo ships that
were procured over a period of several years. A contract with options can be viewed as somewhat similar to an MYP or block buy contract in that a single con
tract is used to procure several years' worth of procurement of a given kind of item. There is, however, a key difference between an MYP or block buy contra
ct and a contract with options: In a contract with options, the service is under no obligation to exercise any of the options, and a service can choose to n
ot exercise an option without having to make a penalty payment to the contractor. In contrast, in an MYP or block buy contract, the service is under an obli
gation to continue implementing the contract beyond the first year, provided that Congress appropriates the necessary funds. If the service chooses to termi
nate an MYP or block buy contract, and does so as a termination for government convenience rather than as a termination for contractor default, then the con
tractor can, under the contract's termination for convenience clause, seek a payment from the government for cost incurred for work that is complete or in p
rocess at the time of termination, and may include the cost of some of the investments made in anticipation of the MYP or block buy contract being fully imp
lemented. The contractor can do this even if the MYP or block buy contract does not elsewhere include a provision for a cancellation penalty. As a result of
this key difference, although a contract with options looks like a multiyear contract, it operates more like a series of annual contracts, and it cannot ach
ieve the kinds of savings that are possible under MYP and BBC. Potential issues for Congress concerning MYP and BBC include whether to use MYP and BBC in th
e future more frequently, less frequently, or about as frequently as they are currently used; and whether to create a permanent statute to govern the use of
BBC, analogous to the permanent statute that governs the use of MYP. Should MYP and BBC in the future be used more frequently, less frequently, or about as
frequently as they are currently used? Supporters of using MYP and BBC more frequently in the future might argue the following: Since MYP and BBC can reduce
procurement costs, making greater use of MYP and BBC can help DOD get more value out of its available procurement funding. This can be particularly important
if DOD's budget in real (i.e., inflation-adjusted) terms remains flat or declines in coming years, as many observers anticipate. The risks of using MYP have
been reduced by Section 811 of the FY2008 National Defense Authorization Act ( H.R. 4986 / P.L. 110-181 of January 28, 2008), which amended 10 U.S.C. 2306b t
o strengthen the process for ensuring that programs proposed for MYP meet certain criteria (see " Permanent Statute Governing MYP "). Since the value of MYP
contracts equated to less than 8% of DOD's procurement budget in FY2012, compared to about 17% of DOD's procurement budget in FY2000, MYP likely could be use
d more frequently without exceeding past experience regarding the share of DOD's procurement budget accounted for by MYP contracts. Supporters of using MYP a
nd BBC less frequently in the future, or at least no more frequently than now, might argue the following: Using MYP and BBC more frequently would further red
uce Congress's and DOD's flexibility for making changes in DOD procurement programs in future years in response to changing strategic or budgetary circumstan
ces. The risks of reducing flexibility in this regard are increased now because of uncertainties in the current strategic environment and because efforts to
reduce federal budget deficits could include reducing DOD spending, which could lead to a reassessment of U.S. defense strategy and associated DOD acquisitio
n programs. Since actual savings from using MYP and BBC rather than annual contracting can be difficult to observe or verify, it is not clear that the financ
ial benefits of using MYP or BBC more frequently in the future would be worth the resulting further reduction in Congress's and DOD's flexibility for making
changes in procurement programs in future years in response to changing strategic or budgetary circumstances. Should Congress create a permanent statute to g
overn the use of BBC, analogous to the permanent statute (10 U.S.C. 2306b) that governs the use of MYP? Supporters of creating a permanent statute to govern
the use of BBC might argue the following: Such a statute could encourage greater use of BBC, and thereby increase savings in DOD procurement programs by givi
ng BBC contracting a formal legal standing and by establishing a clear process for DOD program managers to use in assessing whether their programs might be c
onsidered suitable for BBC. Such a statute could make BBC more advantageous by including a provision that automatically grants EOQ authority to programs usin
g BBC, as well as provisions establishing qualifying criteria and other conditions intended to reduce the risks of using BBC. Opponents of creating a permane
nt statute to govern the use of BBC might argue the following: A key advantage of BBC is that it is not governed by a permanent statute. The lack of such a s
tatute gives DOD and Congress full flexibility in determining when and how to use BBC for programs that may not qualify for MYP, but for which a multiyear co
ntract of some kind might produce substantial savings. Such a statute could encourage DOD program managers to pursue their programs using BBC rather than MYP
. This could reduce discipline in DOD multiyear contracting if the qualifying criteria in the BBC statute are less demanding than the qualifying criteria in
10 U.S.C. 2306b. Should the Coast Guard should begin making use of MYP and BBC? Although the Coast Guard is part of the Department of Homeland Security (DHS)
, the Coast Guard is a military service and a branch of the U.S. Armed Forces at all times (14 U.S.C. 1), and 10 U.S.C. 2306b provides authority for using MY
P not only to DOD, but also to the Coast Guard (and the National Aeronautics and Space Administration as well). In addition, Section 311 of the Frank LoBiond
o Coast Guard Authorization Act of 2018 ( S. 140 / P.L. 115-282 of December 4, 2018) provides permanent authority for the Coast Guard to use block buy contra
cting with EOQ purchases of components in its major acquisition programs. The authority is now codified at 14 U.S.C. 1137. As discussed earlier in this repor
t, the Navy in recent years has made extensive use of MYP and BBC in its ship and aircraft acquisition programs, reducing the collective costs of those progr
ams, the Navy estimates, by billions of dollars. The Coast Guard, like the Navy, procures ships and aircraft. In contrast to the Navy, however, the Coast Gua
rd has never used MYP or BBC in its ship or aircraft acquisition programs. Instead, the Coast has tended to use contracts with options. As discussed earlier,
although a contract with options looks like a multiyear contract, it operates more like a series of annual contracts, and it cannot achieve the kinds of savi
ngs that are possible under MYP and BBC. CRS in recent years has testified and reported on the possibility of using BBC or MYP in Coast Guard ship acquisitio
n programs, particularly the Coast Guard's 25-ship Offshore Patrol Cutter (OPC) program and the Coast Guard's three-ship polar icebreaker program. CRS estima
tes that using multiyear contracting rather than contracts with options for the entire 25-ship OPC program could reduce the cost of the OPC program by about
$1 billion. The OPC program is the Coast Guard's top-priority acquisition program, and it represents a once-in-a-generation opportunity to reduce the acquisi
tion cost of a Coast Guard acquisition program by an estimated $1 billion. CRS also estimates that using BBC for a three-ship polar icebreaker program could
reduce the cost of that program by upwards of $150 million. The Coast Guard has expressed some interest in using BBC in the polar icebreaker program, but its
baseline acquisition strategy for that program, like its current acquisition strategy for the OPC program, is to use a contract with options. As part of its
FY2020 budget submission, the Department of Defense is proposing continued funding for implementing several MYP contracts initiated in fiscal years prior to
FY2020, but it has not highlighted any requests for authority for new MYP or block buy contracts for major acquisition programs that would begin in FY2020. A
ppendix A. Text of 10 U.S.C. 2306b The text of 10 U.S.C. 2306b as of April 29, 2019, is as follows: §2306b. Multiyear contracts: acquisition of property (a)
In General.-To the extent that funds are otherwise available for obligation, the head of an agency may enter into multiyear contracts for the purchase of pro
perty whenever the head of that agency finds each of the following: (1) That the use of such a contract will result in significant savings of the total antic
ipated costs of carrying out the program through annual contracts. (2) That the minimum need for the property to be purchased is expected to remain substanti
ally unchanged during the contemplated contract period in terms of production rate, procurement rate, and total quantities. (3) That there is a reasonable ex
pectation that throughout the contemplated contract period the head of the agency will request funding for the contract at the level required to avoid contra
ct cancellation. (4) That there is a stable design for the property to be acquired and that the technical risks associated with such property are not excessi
ve. (5) That the estimates of both the cost of the contract and the anticipated cost avoidance through the use of a multiyear contract are realistic. (6) In
the case of a purchase by the Department of Defense, that the use of such a contract will promote the national security of the United States. (7) In the case
of a contract in an amount equal to or greater than $500,000,000, that the conditions required by subparagraphs (C) through (F) of subsection (i)(3) will be
met, in accordance with the Secretary's certification and determination under such subsection, by such contract. (b) Regulations.-(1) Each official named in
paragraph (2) shall prescribe acquisition regulations for the agency or agencies under the jurisdiction of such official to promote the use of multiyear cont
racting as authorized by subsection (a) in a manner that will allow the most efficient use of multiyear contracting. (2)(A) The Secretary of Defense shall pr
escribe the regulations applicable to the Department of Defense. (B) The Secretary of Homeland Security shall prescribe the regulations applicable to the Coa
st Guard, except that the regulations prescribed by the Secretary of Defense shall apply to the Coast Guard when it is operating as a service in the Navy. (
) The Administrator of the National Aeronautics and Space Administration shall prescribe the regulations applicable to the National Aeronautics and Space Adm
inistration. (c) Contract Cancellations.-The regulations may provide for cancellation provisions in multiyear contracts to the extent that such provisions ar
e necessary and in the best interests of the United States. The cancellation provisions may include consideration of both recurring and nonrecurring costs of
the contractor associated with the production of the items to be delivered under the contract. (d) Participation by Subcontractors, Vendors, and Suppliers.-I
n order to broaden the defense industrial base, the regulations shall provide that, to the extent practicable- (1) multiyear contracting under subsection (a)
shall be used in such a manner as to seek, retain, and promote the use under such contracts of companies that are subcontractors, vendors, or suppliers; and
(2) upon accrual of any payment or other benefit under such a multiyear contract to any subcontractor, vendor, or supplier company participating in such cont
ract, such payment or benefit shall be delivered to such company in the most expeditious manner practicable. (e) Protection of Existing Authority.-The regula
tions shall provide that, to the extent practicable, the administration of this section, and of the regulations prescribed under this section, shall not be c
arried out in a manner to preclude or curtail the existing ability of an agency- (1) to provide for competition in the production of items to be delivered un
der such a contract; or (2) to provide for termination of a prime contract the performance of which is deficient with respect to cost, quality, or schedule.
(f) Cancellation or Termination for Insufficient Funding.-In the event funds are not made available for the continuation of a contract made under this sectio
n into a subsequent fiscal year, the contract shall be canceled or terminated. The costs of cancellation or termination may be paid from- (1) appropriations
originally available for the performance of the contract concerned; (2) appropriations currently available for procurement of the type of property concerned,
and not otherwise obligated; or (3) funds appropriated for those payments. (g) Contract Cancellation Ceilings Exceeding $100,000,000.-(1) Before any contract
described in subsection (a) that contains a clause setting forth a cancellation ceiling in excess of $100,000,000 may be awarded, the head of the agency conc
erned shall give written notification of the proposed contract and of the proposed cancellation ceiling for that contract to the congressional defense commit
tees, and such contract may not then be awarded until the end of a period of 30 days beginning on the date of such notification. (2) In the case of a contrac
t described in subsection (a) with a cancellation ceiling described in paragraph (1), if the budget for the contract does not include proposed funding for th
e costs of contract cancellation up to the cancellation ceiling established in the contract, the head of the agency concerned shall, as part of the certifica
tion required by subsection (i)(1)(A),1 give written notification to the congressional defense committees of- (A) the cancellation ceiling amounts planned fo
r each program year in the proposed multiyear procurement contract, together with the reasons for the amounts planned; (B) the extent to which costs of contr
act cancellation are not included in the budget for the contract; and (C) a financial risk assessment of not including budgeting for costs of contract cancel
lation. (h) Defense Acquisitions of Weapon Systems.-In the case of the Department of Defense, the authority under subsection (a) includes authority to enter
into the following multiyear contracts in accordance with this section: (1) A multiyear contract for the purchase of a weapon system, items and services asso
ciated with a weapon system, and logistics support for a weapon system. (2) A multiyear contract for advance procurement of components, parts, and materials
necessary to the manufacture of a weapon system, including a multiyear contract for such advance procurement that is entered into in order to achieve economi
c-lot purchases and more efficient production rates. (i) Defense Acquisitions Specifically Authorized by Law.-(1) In the case of the Department of Defense, a
multiyear contract in an amount equal to or greater than $500,000,000 may not be entered into under this section unless the contract is specifically authoriz
ed by law in an Act other than an appropriations Act. (2) In submitting a request for a specific authorization by law to carry out a defense acquisition prog
ram using multiyear contract authority under this section, the Secretary of Defense shall include in the request the following: (A) A report containing preli
minary findings of the agency head required in paragraphs (1) through (6) of subsection (a), together with the basis for such findings. (B) Confirmation that
the preliminary findings of the agency head under subparagraph (A) were supported by a preliminary cost analysis performed by the Director of Cost Assessment
and Program Evaluation. (3) A multiyear contract may not be entered into under this section for a defense acquisition program that has been specifically auth
orized by law to be carried out using multiyear contract authority unless the Secretary of Defense certifies in writing, not later than 30 days before entry
into the contract, that each of the following conditions is satisfied: (A) The Secretary has determined that each of the requirements in paragraphs (1) throu
gh (6) of subsection (a) will be met by such contract and has provided the basis for such determination to the congressional defense committees. (B) The Secr
etary's determination under subparagraph (A) was made after completion of a cost analysis conducted on the basis of section 2334(e)(2) 1 of this title, and t
he analysis supports the determination. (C) The system being acquired pursuant to such contract has not been determined to have experienced cost growth in ex
cess of the critical cost growth threshold pursuant to section 2433(d) of this title within 5 years prior to the date the Secretary anticipates such contract
(or a contract for advance procurement entered into consistent with the authorization for such contract) will be awarded. (D) A sufficient number of end item
s of the system being acquired under such contract have been delivered at or within the most current estimates of the program acquisition unit cost or procur
ement unit cost for such system to determine that current estimates of such unit costs are realistic. (E) During the fiscal year in which such contract is to
be awarded, sufficient funds will be available to perform the contract in such fiscal year, and the future-years defense program for such fiscal year will in
clude the funding required to execute the program without cancellation. (F) The contract is a fixed price type contract. (G) The proposed multiyear contract
provides for production at not less than minimum economic rates given the existing tooling and facilities. (4) If for any fiscal year a multiyear contract
be entered into under this section is authorized by law for a particular procurement program and that authorization is subject to certain conditions establis
hed by law (including a condition as to cost savings to be achieved under the multiyear contract in comparison to specified other contracts) and if it appear
s (after negotiations with contractors) that such savings cannot be achieved, but that significant savings could nevertheless be achieved through the use of
a multiyear contract rather than specified other contracts, the President may submit to Congress a request for relief from the specified cost savings that mu
st be achieved through multiyear contracting for that program. Any such request by the President shall include details about the request for a multiyear cont
ract, including details about the negotiated contract terms and conditions. (5)(A) The Secretary may obligate funds for procurement of an end item under a mul
tiyear contract for the purchase of property only for procurement of a complete and
usable end item. (B) The Secretary may obligate funds appropriated for any fiscal year for advance procurement under a contract for the purchase of property o
nly for the procurement of those long-lead items necessary in order to meet a planned delivery schedule for complete major end items that are programmed under
the contract to be acquired with funds appropriated for a subsequent fiscal year (including an economic order quantity of such long-lead items when authorized
by law). (6) The Secretary may make the certification under paragraph (3) notwithstanding the fact that one or more of the conditions of such certification ar
e not met, if the Secretary determines that, due to exceptional circumstances, proceeding with a multiyear contract under this section is in the best interest
of the Department of Defense and the Secretary provides the basis for such determination with the certification. (7) The Secretary may not delegate the author
ity to make the certification under paragraph (3) or the determination under paragraph (6) to an official below the level of Under Secretary of Defense for Ac
quisition, Technology, and Logistics. (j) Defense Contract Options for Varying Quantities.-The Secretary of Defense may instruct the Secretary of the military
department concerned to incorporate into a proposed multiyear contract negotiated priced options for varying the quantities of end items to be procured over th
e period of the contract. (k) Multiyear Contract Defined.-For the purposes of this section, a multiyear contract is a contract for the purchase of property for
more than one, but not more than five, program years. Such a contract may provide that performance under the contract during the second and subsequent years of
the contract is contingent upon the appropriation of funds and (if it does so provide) may provide for a cancellation payment to be made to the contractor if s
uch appropriations are not made. (l) Various Additional Requirements With Respect to Multiyear Defense Contracts.-(1)(A) The head of an agency may not initiate
a contract described in subparagraph (B) unless the congressional defense committees are notified of the proposed contract at least 30 days in advance of the aw
ard of the proposed contract. (B) Subparagraph (A) applies to the following contracts: (i) A multiyear contract- (I) that employs economic order quantity procur
ement in excess of $20,000,000 in any one year of the contract; or (II) that includes an unfunded contingent liability in excess of $20,000,000. (ii) Any contra
ct for advance procurement leading to a multiyear contract that employs economic order quantity procurement in excess of $20,000,000 in any one year. (2) The hea
d of an agency may not initiate a multiyear contract for which the economic order quantity advance procurement is not funded at least to the limits of the Govern
ment's liability. (3) The head of an agency may not initiate a multiyear procurement contract for any system (or component thereof) if the value of the multiyear
contract would exceed $500,000,000 unless authority for the contract is specifically provided in an appropriations Act. (4) Each report required by paragraph (5)
with respect to a contract (or contract extension) shall contain the following: (A) The amount of total obligational authority under the contract (or contract ex
tension) and the percentage that such amount represents of- (i) the applicable procurement account; and (ii) the agency procurement total. (B) The amount of tota
l obligational authority under all multiyear procurements of the agency concerned (determined without regard to the amount of the multiyear contract (or contract
extension)) under multiyear contracts in effect at the time the report is submitted and the percentage that such amount represents of- (i) the applicable procure
ment account; and (ii) the agency procurement total. (C) The amount equal to the sum of the amounts under subparagraphs (A) and (B), and the percentage that suc
h amount represents of- (i) the applicable procurement account; and (ii) the agency procurement total. (D) The amount of total obligational authority under all
Department of Defense multiyear procurements (determined without regard to the amount of the multiyear contract (or contract extension)), including any multiye
ar contract (or contract extension) that has been authorized by the Congress bu
t not yet entered into, and the percentage that such amount represents of the procurement accounts of the Department of Defense treated in the aggregate. (5) T
he head of an agency may not enter into a multiyear contract (or extend an existing multiyear contract), the value of which would exceed $500,000,000 (when en
tered into or when extended, as the case may be), until the Secretary of Defense submits to the congressional defense committees a report containing the info
rmation described in paragraph (4) with respect to the contract (or contract extension). (6) The head of an agency may not terminate a multiyear procurement c
ontract until 10 days after the date on which notice of the proposed termination is provided to the congressional defense committees. (7) The execution of mul
tiyear contracting authority shall require the use of a present value analysis to determine lowest cost compared to an annual procurement. (8) This subsection
does not apply to the National Aeronautics and Space Administration or to the Coast Guard. (9) In this subsection: (A) The term "applicable procurement accoun
t" means, with respect to a multiyear procurement contract (or contract extension), the appropriation account from which payments to execute the contract will
be made. (B) The term "agency procurement total" means the procurement accounts of the agency entering into a multiyear procurement contract (or contract exten
sion) treated in the aggregate. (m) Increased Funding and Reprogramming Requests.-Any request for increased funding for the procurement of a major system under
a multiyear contract authorized under this section shall be accompanied by an explanation of how the request for increased funding affects the determinations ma
de by the Secretary under subsection (i). Appendix B. Programs Approved for MYP in Annual DOD Appropriations Acts Since FY1990 This appendix presents, in two t
ables, programs approved for MYP in annual DOD appropriations acts since FY1990. Tab
le B-1 covers FY2011 to the present, and Table B-2 covers FY1990 through FY2010.
""".replace(
    "\n", ""
)

keywords = ["MYP", "DOD", "BBC", "10 U.S.C. 2306b."]

occurances = {}

for keyword in keywords:
    oss = []
    for match in re.finditer(keyword.lower(), GOVREPORT_CONTENT.lower()):
        oss.append(match.start())
    occurances[keyword] = oss

tokens_location_512_window = len(GOVREPORT_CONTENT) - len(
    " ".join(GOVREPORT_CONTENT.split()[-256:])
)
tokens_location_1024_window = len(GOVREPORT_CONTENT) - len(
    " ".join(GOVREPORT_CONTENT.split()[-512:])
)


setup_seaborn(axis_below=True)

bins = np.arange(0, len(GOVREPORT_CONTENT), len(GOVREPORT_CONTENT) // 100)

plt.figure(figsize=(8, 1))
data = []
for keyword in keywords:
    data.append(occurances[keyword])
plt.hist(
    data,
    label=keywords,
    bins=100,
    stacked=True,
    color=["#0091FF", "#FF9100", "#03FF92", "#FF0080"],
)
plt.legend(loc="center right", bbox_to_anchor=(0.935, 0.72), ncols=2, fontsize=6)
plt.axvline(tokens_location_512_window, color="#fc3897")
plt.axvline(tokens_location_1024_window, color="#fc4138")
plt.annotate(
    "512 Token Window",
    [tokens_location_512_window - 600, 4.0],
    color="#fc3897",
    backgroundcolor="white",
    horizontalalignment="right",
    fontsize=7,
    fontweight=700,
)
plt.annotate(
    "1024 Token Window",
    [tokens_location_1024_window - 700, 1.5],
    color="#fc4138",
    backgroundcolor="white",
    horizontalalignment="right",
    fontsize=7,
    fontweight=700,
)
plt.xlim(0, len(GOVREPORT_CONTENT))
plt.ylabel("Freq.")
plt.xlabel("Document Location (Characters)")
# plt.title('Keyword Occurance Histogram (Stacked)')

root = "saves/rouge"
os.makedirs(root, exist_ok=True)
path = os.path.join(root, "hist")
plt.savefig(path + ".png", bbox_inches="tight", pad_inches=0.1)
plt.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.1)
plt.savefig(path + ".svg", bbox_inches="tight", pad_inches=0.1)
print("saved", path + ".png")
