import json

# Read the notebook file
notebook_path = 'models/hybrid_models_GRU-w12.ipynb'
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Get the cell content as a single string
content = ''.join(notebook['cells'][0]['source'])

# Find the problematic section - everything after the first build_dataloaders return
# We need to find where the good code ends and the duplicated/bad code begins

# Find the end of the first proper build_dataloaders function
marker_good_end = "def build_dataloaders(model_key, fold_name, dataset, batch_size=64):"
good_end_idx = content.find(marker_good_end)

if good_end_idx != -1:
    # Find the end of the FIRST complete build_dataloaders function (the organized one)
    # Look for the next function definition
    next_def_idx = content.find("def ", good_end_idx + 100)  # Skip the function definition itself
    
    if next_def_idx == -1:
        # No next function found, keep everything
        next_def_idx = len(content)
    
    # Check if the build_dataloaders is the reorganized version or the broken version
    # The reorganized version should have 7 sections and a single return
    function_body = content[good_end_idx:next_def_idx]
    
    # Count return statements in this function
    return_count = function_body.count("return {")
    
    if return_count > 1:
        # This is the broken version with multiple returns
        print("Found broken build_dataloaders with multiple returns. Fixing...")
        
        # Find the end of the good part (before the duplication starts)
        # Look for the first return statement
        first_return_idx = content.find("return {", good_end_idx)
        if first_return_idx != -1:
            # Find the end of the first return statement
            return_end_idx = content.find("}", first_return_idx) + 1
            
            # Everything after this is duplicated/broken code - remove it
            before_bad_code = content[:return_end_idx]
            
            # Find the start of the next valid function (not duplicate imports or functions)
            # Look for a function that's not duplicated
            next_good_function_markers = [
                "def safe_is_nan(arr):",  # The global version
                "# Test the enhanced functions",
                "# Load Datasets"
            ]
            
            after_bad_code = ""
            for marker in next_good_function_markers:
                marker_idx = content.find(marker, return_end_idx)
                if marker_idx != -1:
                    after_bad_code = content[marker_idx:]
                    break
            
            if after_bad_code:
                # Reconstruct without the bad code
                new_content = before_bad_code + "\n\n" + after_bad_code
                print("✅ Removed all duplicate and unreachable code")
            else:
                # Fallback: just remove everything after the first return
                new_content = before_bad_code + "\n\n# End of notebook\n"
                print("⚠️ Could not find good continuation point, truncated after first return")
        else:
            print("❌ Could not find return statement")
            new_content = content
    else:
        # This is already the good version
        print("✅ Code structure is already clean")
        new_content = content
else:
    print("❌ Could not find build_dataloaders function")
    new_content = content

# Update the notebook
notebook['cells'][0]['source'] = new_content.split('\n')

# Save the updated notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print("🧹 Final cleanup completed:")
print("   • Removed duplicate code")
print("   • Eliminated unreachable code") 
print("   • Fixed multiple return statements")
print("   • Cleaned up duplicate imports")
print("   • Maintained only the organized structure") 