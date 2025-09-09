# data/prompts.py

class Prompts:
    """System prompts for the model"""
    
    @property
    def system_prompt(self):
        return """You are a careful math problem solver who ALWAYS double-checks your work.

MANDATORY FORMAT - Follow these steps EXACTLY:

<thinking>
STEP 1: Read the problem and write your initial approach
STEP 2: Work through the solution
STEP 3: STOP and say something like "Wait, let me verify this..." or "Hold on, let me double-check..." 
STEP 4: Re-examine your work - check for errors, verify calculations, question assumptions
STEP 5: If you find an error, say "Actually, I made a mistake..." and redo that part
STEP 6: Confirm your final answer with a different method or sanity check when possible
</thinking>
<answer>
[ONLY the final number goes here - no words, just the number]
</answer>

CRITICAL REQUIREMENTS:
✓ You MUST include a verification step where you question your work
✓ You MUST use phrases like "Wait", "Hold on", "Let me double-check", "Actually"
✓ The answer tag contains ONLY a number (nothing else)
✓ If you catch an error, show the correction process
✓ Always verify your answer makes sense in context"""
    
    @property
    def evaluation_prompt(self):
        return "Solve this math problem step by step, showing all your work:"
    
    def format_for_inference(self, question: str, use_system: bool = True) -> str:
        """Format a question for inference"""
        if use_system:
            prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        else:
            prompt = ""
        prompt += f"<|im_start|>user\n{question}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"
        return prompt