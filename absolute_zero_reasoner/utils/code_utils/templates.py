from typing import List


RUN_CODE_TEMPLATE = """{code}
repr(f({inputs}))"""

RUN_CODE_TEMPLATE_REPR = """{code}
print('<FINAL_REPR_SYMBOL>', repr(f({inputs})))"""

VALIDATE_CODE_TEMPLATE = """{code}
repr(f({inputs}))"""

VALIDATE_CODE_TEMPLATE_REPR = """{code}
print('<FINAL_REPR_SYMBOL>', repr(f({inputs})))"""

EVAL_INPUT_PREDICTION_TEMPLATE = """{code}
{gold_output} == f({agent_input})"""

EVAL_INPUT_PREDICTION_TEMPLATE_REPR = """{code}
print('<FINAL_REPR_SYMBOL>', repr({gold_output} == f({agent_input})))"""

EVAL_OUTPUT_PREDICTION_TEMPLATE = """{code}
eval({gold_output}) == eval({agent_output})"""

EVAL_OUTPUT_PREDICTION_TEMPLATE_REPR = """{code}
print('<FINAL_REPR_SYMBOL>', repr(eval({gold_output}) == eval({agent_output})))"""

CHECK_DETERMINISM_TEMPLATE = """{code}
returns = f({inputs})
if returns != f({inputs}):
    raise Exception('Non-deterministic code')
repr(returns)"""

CHECK_DETERMINISM_TEMPLATE_REPR = """{code}
returns = f({inputs})
if returns != f({inputs}):
    raise Exception('Non-deterministic code')
print('<FINAL_REPR_SYMBOL>', repr(returns))"""

def EVAL_K_INPUT_PREDICTION_TEMPLATE(code: str, gold_output: str, k_agent_inputs: List[str], repr_output: bool = False):
    output_string = f"""{code}
acc_list = []"""
    for inp in k_agent_inputs:
        output_string += f"""\ntry:
    acc_list.append({gold_output} == f({inp}))
except:
    acc_list.append(False)"""
    # then compute the mean of the list
    if repr_output:
        output_string += """\nprint('<FINAL_REPR_SYMBOL>', repr(acc_list))"""
    else:
        output_string += """\nacc_list"""
    return output_string

def EVAL_K_OUTPUT_PREDICTION_TEMPLATE(code: str, gold_output: str, k_agent_outputs: List[str], repr_output: bool = False):
    output_string = f"""{code}
acc_list = []"""
    for out in k_agent_outputs:
        output_string += f"""\ntry:
    acc_list.append({gold_output} == {out})
except:
    acc_list.append(False)"""
    # then compute the mean of the list
    if repr_output:
        output_string += """\nprint('<FINAL_REPR_SYMBOL>', repr(acc_list))"""
    else:
        output_string += """\nacc_list"""
    return output_string
