"""Plain-English intent detection for the chatbot (no ML needed): what kind of question is this?"""
import re

URGENT = re.compile(r"chest pain|can'?t breathe|cannot breathe|trouble breathing|difficulty breathing|"
                    r"short(ness)?[ -]of[ -]breath|short[- ]?breath(ed)?|out of breath|catch my breath|gasping|"
                    r"coughing (up )?blood|faint(ed|ing)|passed out|blue lips|severe pain|emergency", re.I)

WHOLE_REPORT = re.compile(r"explain.*(report|result)|summar|what does (my|the) report|overall|in simple|"
                          r"why.*flag|what (was|were|is|are) flagged|which .*flagged", re.I)

# "am I healthy?", "what happens to me?", "is this serious?", "do I have a problem?" ...
HEALTH_STATUS = re.compile(
    r"\bam i (healthy|ok|okay|fine|normal|sick|safe|dying|going to die)|\bis (everything|all|it|this) (ok|okay|fine|normal|bad|serious|dangerous)"
    r"|what('s| is) wrong with me|what (will |is going to )?happens? (to|with) me|what does (this|it|that) mean for me"
    r"|\bdo i have (a |any |some )?(\w+ ){0,3}(problem|disease|illness|condition|cancer|infection|issue)"
    r"|\bshould i (worry|be worried|be concerned)|\bhow bad\b|\bwill i be (ok|okay|fine|alright)|\bam i (in danger|at risk)", re.I)

# "I have a cough", "I feel dizzy", "I'm facing shortness of breath"...
SYMPTOMS = re.compile(
    r"\b(symptoms?|cough(ing)?|breath(less|ing)?|short[- ]?breath(ed)?|wheez(e|ing)|chest (pain|tightness|pressure)|fever|"
    r"pain(s|ful)?|tired(ness)?|fatigue|dizz(y|iness)|nausea|sweat(ing|s)?|palpitations?|swelling|weakness|"
    r"headache|vomit(ing)?|chills|unwell|sick|ill)\b", re.I)


def flags(question: str) -> dict:
    return {"urgent": bool(URGENT.search(question)), "whole": bool(WHOLE_REPORT.search(question)),
            "health": bool(HEALTH_STATUS.search(question)), "symptoms": bool(SYMPTOMS.search(question))}
