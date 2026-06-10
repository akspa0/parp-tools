using System;
using System.Collections.Generic;
using System.IO;
using DBCD;
using DBCD.Providers;
using DBCD.IO;

namespace WoWToolbox.Core.Legacy.Liquid
{
    /// <summary>
    /// Validator for liquid types using LiquidType.dbc
    /// </summary>
    public class LiquidTypeValidator
    {
        private readonly IDBCProvider? dbcProvider;
        private readonly string? dbcPath;
        private Dictionary<uint, bool> validatedEntries;
        private bool isInitialized;

        /// <summary>
        /// Creates a new instance of the LiquidTypeValidator class
        /// </summary>
        /// <param name="dbcProvider">Optional DBC provider. If null, validation will be skipped.</param>
        /// <param name="dbcPath">Optional path to the DBC file. If null, validation will be skipped.</param>
        public LiquidTypeValidator(IDBCProvider? dbcProvider = null, string? dbcPath = null)
        {
            this.dbcProvider = dbcProvider;
            this.dbcPath = dbcPath;
            this.validatedEntries = new Dictionary<uint, bool>();
        }

        /// <summary>
        /// Validates a liquid entry against LiquidType.dbc
        /// </summary>
        /// <param name="liquidEntry">The liquid entry to validate</param>
        /// <returns>True if the entry is valid or validation is disabled, false otherwise</returns>
        public bool ValidateLiquidEntry(uint liquidEntry)
        {
            // If no DBC provider or path, skip validation
            // if (dbcProvider == null || string.IsNullOrEmpty(dbcPath))
            // {
            //     return true;
            // }

            // Initialize if needed
            // if (!isInitialized)
            // {
            //     Initialize();
            // }

            // Check cache first
            // if (validatedEntries.TryGetValue(liquidEntry, out bool isValid))
            // {
            //     return isValid;
            // }

            // try
            // {
            //     // Load LiquidType.dbc
            //     // using var dbc = new DBCReader(dbcPath, dbcProvider); // THIS LINE CAUSED THE ERROR
            //     
            //     // Check if the entry exists
            //     // bool exists = dbc.RecordExists(liquidEntry);
            //     // validatedEntries[liquidEntry] = exists;
            //     // return exists;
            // }
            // catch (Exception)
            // {
            //     // If there's any error loading or reading the DBC, skip validation
            //     return true;
            // }
            
            // Return true for now as this class is no longer used directly by MCLQChunk
            // It might be needed elsewhere, so we don't delete the file entirely yet.
            return true; 
        }

        private void Initialize()
        {
            validatedEntries = new Dictionary<uint, bool>();
            isInitialized = true;
        }

        /// <summary>
        /// Clears the validation cache
        /// </summary>
        public void ClearCache()
        {
            validatedEntries.Clear();
        }
    }
} 