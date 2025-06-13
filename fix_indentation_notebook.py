#!/usr/bin/env python3
"""Script para corregir la indentación en el notebook base_models_Conv_STHyMOUNTAIN.ipynb"""

import json

def fix_indentation():
    # Leer el notebook
    with open('models/base_models_Conv_STHyMOUNTAIN.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Buscar la primera celda de código
    for cell_idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            
            # Buscar si esta celda contiene el código problemático
            if 'class TrainingMonitor(Callback):' in source and 'plt.tight_layout()' in source:
                # Dividir en líneas
                lines = source.split('\n')
                fixed_lines = []
                in_on_epoch_end = False
                
                for i, line in enumerate(lines):
                    # Detectar cuando entramos en on_epoch_end
                    if 'def on_epoch_end(self, epoch, logs=None):' in line:
                        in_on_epoch_end = True
                        fixed_lines.append(line)
                        continue
                    
                    # Detectar cuando salimos del método (siguiente def o class)
                    if in_on_epoch_end and line.strip() and not line.startswith(' ') and not line.strip().startswith('#'):
                        in_on_epoch_end = False
                    
                    # Si estamos dentro de on_epoch_end y la línea tiene indentación incorrecta
                    if in_on_epoch_end:
                        # Líneas que deben tener indentación de 8 espacios
                        if line.strip() in ['plt.tight_layout()', 
                                           'plt.subplots_adjust(hspace=0.4, wspace=0.3)',
                                           'display(fig)',
                                           'plt.close()']:
                            fixed_lines.append(' ' * 8 + line.strip())
                        elif line.strip().startswith('# Mostrar métricas actuales'):
                            fixed_lines.append(' ' * 8 + line.strip())
                        elif line.strip().startswith('print(f"\\n📊 Época'):
                            fixed_lines.append(' ' * 8 + line.strip())
                        elif line.strip().startswith('print(f"   •'):
                            fixed_lines.append(' ' * 8 + line.strip())
                        elif line.strip().startswith('# Mostrar mejora'):
                            fixed_lines.append(' ' * 8 + line.strip())
                        elif line.strip() == 'if len(self.val_losses) > 1:' and i > 0 and 'Mostrar mejora' in lines[i-1]:
                            fixed_lines.append(' ' * 8 + line.strip())
                        elif line.strip().startswith('improvement = (self.val_losses[-2]') and i > 0 and 'if len(self.val_losses) > 1:' in lines[i-1]:
                            fixed_lines.append(' ' * 12 + line.strip())
                        elif line.strip().startswith('print(f"   • Mejora:') and i > 0 and 'improvement = ' in lines[i-1]:
                            fixed_lines.append(' ' * 12 + line.strip())
                        else:
                            fixed_lines.append(line)
                    else:
                        fixed_lines.append(line)
                
                # Actualizar el source de la celda
                new_source = '\n'.join(fixed_lines)
                cell['source'] = new_source.split('\n')
                
                # Asegurarse de que cada línea termine con \n excepto la última
                for j in range(len(cell['source']) - 1):
                    if not cell['source'][j].endswith('\n'):
                        cell['source'][j] += '\n'
                
                # Guardar el notebook corregido
                with open('models/base_models_Conv_STHyMOUNTAIN_fixed.ipynb', 'w', encoding='utf-8') as f:
                    json.dump(notebook, f, indent=1, ensure_ascii=False)
                
                print("✅ Notebook corregido guardado como base_models_Conv_STHyMOUNTAIN_fixed.ipynb")
                return True
    
    print("❌ No se encontró la celda con el problema")
    return False

if __name__ == "__main__":
    fix_indentation() 